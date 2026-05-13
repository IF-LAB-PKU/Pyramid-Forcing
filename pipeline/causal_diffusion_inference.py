from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from headkv import AdaptiveKVCache, HeadKVCache, HeadKVConfig
from pipeline.headkv_config import HeadKVPipelineConfig


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.use_headkv = getattr(args, "use_headkv", False)
        self.headkv_config = HeadKVPipelineConfig.from_args(args, frame_seq_length=self.frame_seq_length)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            context_len = 0
            if self.use_headkv and self.headkv_config.headkv_is_i2v and initial_latent is not None:
                context_len = self.headkv_config.headkv_context_len
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                context_len=context_len
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
                self.crossattn_cache_pos[block_index]["prompt_v"] = None
                self.crossattn_cache_neg[block_index]["prompt_v"] = None
            # reset kv cache
            if self.use_headkv:
                for cache in self.kv_cache_pos:
                    cache.reset()
                for cache in self.kv_cache_neg:
                    cache.reset()
            else:
                for block_index in range(len(self.kv_cache_pos)):
                    self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="noisy",
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                    cache_update_mode="noisy",
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                cache_update_mode="clean",
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
                cache_update_mode="clean",
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, context_len=0):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if self.use_headkv:
            hc = self.headkv_config
            num_layers = self.generator.model.num_layers
            num_heads = self.generator.model.num_heads
            head_dim = self.generator.model.dim // num_heads
            if self.local_attn_size != -1:
                base_capacity_tokens = self.local_attn_size * self.frame_seq_length
            else:
                base_capacity_tokens = 32760
            default_capacity = hc.headkv_default_capacity or base_capacity_tokens
            config = HeadKVConfig(
                hc.headkv_config_path,
                num_layers=num_layers,
                num_heads=num_heads,
                default_capacity=default_capacity,
                strategy_reduction_factor=hc.headkv_strategy_factor,
                code_map=hc.headkv_code_map,
                head_type_csv_path=hc.headkv_policy_csv_path,
                drop_heads_csv_path=hc.headkv_drop_heads_csv_path,
                soft_ablate_heads_csv_path=hc.headkv_soft_ablate_csv_path,
                af_policy_enabled=hc.headkv_af_policy_enabled,
                af_csv_path=hc.headkv_af_csv_path,
                af_group_dir=hc.headkv_af_group_dir,
                af_manifest_path=hc.headkv_af_manifest_path,
                frame_seq_length=hc.headkv_frame_seq_length,
            )
            self.kv_cache_pos = [
                (
                    AdaptiveKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.headkv_is_i2v,
                        context_len=context_len,
                        sink_len=hc.headkv_sink_tokens,
                        tail_len=hc.headkv_dynamic_capacity,
                        ivc_ratio=hc.ivc_ratio,
                        semantic_ratio=hc.semantic_ratio,
                        trajectory_ratio=hc.trajectory_ratio,
                        trajectory_weight=hc.trajectory_weight,
                        history_frame_quota=hc.history_frame_quota,
                        history_quota_ivc_ratio=hc.history_quota_ivc_ratio,
                        post_train_stabilize_t=hc.post_train_stabilize_t,
                        post_train_trajectory_scale=hc.post_train_trajectory_scale,
                        post_train_history_ivc_ratio=hc.post_train_history_ivc_ratio,
                        update_interval=hc.update_interval,
                        seed_ratio=hc.semantic_seed_ratio,
                        sink_grid_decoupling=hc.sink_grid_decoupling,
                        decoupled_sink_tokens=hc.decoupled_sink_tokens,
                        decoupled_sink_time_lag=hc.decoupled_sink_time_lag,
                        sink_time_mapping_mode=hc.headkv_dynamic_rope_mode,
                        sink_time_clamp_min=hc.sink_time_clamp_min,
                        sink_time_clamp_max=hc.sink_time_clamp_max,
                        history_time_mapping_mode=hc.history_time_mapping_mode,
                        history_relative_t_max=hc.history_relative_t_max,
                        history_time_soft_factor=hc.history_time_soft_factor,
                        use_osc_frame_mode=hc.cyclic_enabled,
                        phase_period=hc.cyclic_period,
                        phase_bucket_capacity_frames=hc.cyclic_bucket_cap,
                        local_tail_frames=hc.headkv_recent_frames,
                        phase_sink_for_osc_only=hc.cyclic_osc_only,
                        phase_sink_dynamic_rope=hc.cyclic_dynamic_rope,
                        use_osc_lag_mode=hc.lag_enabled,
                        osc_lag_offsets_frames=hc.headkv_lag_offsets,
                        osc_lag_history_frames=hc.headkv_lag_history,
                        osc_lag_dynamic_rope=hc.lag_dynamic_rope,
                        disable_first_sink_for_osc_heads=hc.headkv_disable_osc_sink,
                        use_stable_head_policies=hc.headkv_stable_policy_enabled,
                        stable_sink_frames=hc.headkv_stable_sink_frames,
                        osc_sink_frames=hc.headkv_osc_sink_frames,
                        stable_recent_frames=hc.headkv_stable_recent_frames,
                        use_af_head_policies=hc.headkv_af_policy_enabled,
                        af_recent_frames_map=hc.headkv_af_recent_frames_map,
                        af_phase_bucket_map=hc.headkv_af_phase_bucket_map,
                        af_lag_offsets_map=hc.headkv_af_lag_offsets_map,
                        af_sink_frames_map=hc.headkv_af_sink_frames_map,
                        af_stride_enabled_map=hc.headkv_af_stride_enabled_map,
                        label_recent_frames_map=hc.headkv_label_recent_frames_map,
                        label_phase_bucket_map=hc.headkv_label_phase_bucket_map,
                        label_lag_offsets_map=hc.headkv_label_lag_offsets_map,
                        label_sink_frames_map=hc.headkv_label_sink_frames_map,
                        label_stride_enabled_map=hc.headkv_label_stride_enabled_map,
                        capture_frame_id_mode=hc.headkv_capture_frame_id_mode,
                        readout_cache_enabled=hc.headkv_readout_cache_enabled,
                        prompt_value_cache_enabled=hc.headkv_prompt_v_cache_enabled,
                    )
                    if hc.use_adaptive_headkv else
                    HeadKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.headkv_is_i2v,
                        context_len=context_len,
                        frame_seq_length=hc.headkv_frame_seq_length,
                        prompt_value_cache_enabled=hc.headkv_prompt_v_cache_enabled,
                    )
                )
                for layer_idx in range(num_layers)
            ]
            self.kv_cache_neg = [
                (
                    AdaptiveKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.headkv_is_i2v,
                        context_len=context_len,
                        sink_len=hc.headkv_sink_tokens,
                        tail_len=hc.headkv_dynamic_capacity,
                        ivc_ratio=hc.ivc_ratio,
                        semantic_ratio=hc.semantic_ratio,
                        trajectory_ratio=hc.trajectory_ratio,
                        trajectory_weight=hc.trajectory_weight,
                        history_frame_quota=hc.history_frame_quota,
                        history_quota_ivc_ratio=hc.history_quota_ivc_ratio,
                        post_train_stabilize_t=hc.post_train_stabilize_t,
                        post_train_trajectory_scale=hc.post_train_trajectory_scale,
                        post_train_history_ivc_ratio=hc.post_train_history_ivc_ratio,
                        update_interval=hc.update_interval,
                        seed_ratio=hc.semantic_seed_ratio,
                        sink_grid_decoupling=hc.sink_grid_decoupling,
                        decoupled_sink_tokens=hc.decoupled_sink_tokens,
                        decoupled_sink_time_lag=hc.decoupled_sink_time_lag,
                        sink_time_mapping_mode=hc.headkv_dynamic_rope_mode,
                        sink_time_clamp_min=hc.sink_time_clamp_min,
                        sink_time_clamp_max=hc.sink_time_clamp_max,
                        history_time_mapping_mode=hc.history_time_mapping_mode,
                        history_relative_t_max=hc.history_relative_t_max,
                        history_time_soft_factor=hc.history_time_soft_factor,
                        use_osc_frame_mode=hc.cyclic_enabled,
                        phase_period=hc.cyclic_period,
                        phase_bucket_capacity_frames=hc.cyclic_bucket_cap,
                        local_tail_frames=hc.headkv_recent_frames,
                        phase_sink_for_osc_only=hc.cyclic_osc_only,
                        phase_sink_dynamic_rope=hc.cyclic_dynamic_rope,
                        use_osc_lag_mode=hc.lag_enabled,
                        osc_lag_offsets_frames=hc.headkv_lag_offsets,
                        osc_lag_history_frames=hc.headkv_lag_history,
                        osc_lag_dynamic_rope=hc.lag_dynamic_rope,
                        disable_first_sink_for_osc_heads=hc.headkv_disable_osc_sink,
                        use_stable_head_policies=hc.headkv_stable_policy_enabled,
                        stable_sink_frames=hc.headkv_stable_sink_frames,
                        osc_sink_frames=hc.headkv_osc_sink_frames,
                        stable_recent_frames=hc.headkv_stable_recent_frames,
                        use_af_head_policies=hc.headkv_af_policy_enabled,
                        af_recent_frames_map=hc.headkv_af_recent_frames_map,
                        af_phase_bucket_map=hc.headkv_af_phase_bucket_map,
                        af_lag_offsets_map=hc.headkv_af_lag_offsets_map,
                        af_sink_frames_map=hc.headkv_af_sink_frames_map,
                        af_stride_enabled_map=hc.headkv_af_stride_enabled_map,
                        label_recent_frames_map=hc.headkv_label_recent_frames_map,
                        label_phase_bucket_map=hc.headkv_label_phase_bucket_map,
                        label_lag_offsets_map=hc.headkv_label_lag_offsets_map,
                        label_sink_frames_map=hc.headkv_label_sink_frames_map,
                        label_stride_enabled_map=hc.headkv_label_stride_enabled_map,
                        capture_frame_id_mode=hc.headkv_capture_frame_id_mode,
                        readout_cache_enabled=hc.headkv_readout_cache_enabled,
                        prompt_value_cache_enabled=hc.headkv_prompt_v_cache_enabled,
                    )
                    if hc.use_adaptive_headkv else
                    HeadKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.headkv_is_i2v,
                        context_len=context_len,
                        frame_seq_length=hc.headkv_frame_seq_length,
                        prompt_value_cache_enabled=hc.headkv_prompt_v_cache_enabled,
                    )
                )
                for layer_idx in range(num_layers)
            ]
            # Soft ablation controls are runtime knobs on cache objects.
            for cache in self.kv_cache_pos:
                cache.soft_ablate_region = str(hc.headkv_soft_ablate_region)
                cache.soft_ablate_scale = float(hc.headkv_soft_ablate_scale)
            for cache in self.kv_cache_neg:
                cache.soft_ablate_region = str(hc.headkv_soft_ablate_region)
                cache.soft_ablate_scale = float(hc.headkv_soft_ablate_scale)
        else:
            kv_cache_pos = []
            kv_cache_neg = []
            if self.local_attn_size != -1:
                # Use the local attention size to compute the KV cache size
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Use the default KV cache size
                kv_cache_size = 32760

            for _ in range(self.num_transformer_blocks):
                kv_cache_pos.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
                kv_cache_neg.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })

            self.kv_cache_pos = kv_cache_pos  # always store the clean cache
            self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
                "prompt_v": None,
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
                "prompt_v": None,
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
