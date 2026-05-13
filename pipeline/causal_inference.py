from typing import List, Optional
import os
import torch
from tqdm import tqdm

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pyramidkv import AdaptiveKVCache, PyramidKVCache, PyramidKVConfig
from pyramidkv import build_compositions
from pyramidkv._mega_cache import build_mega_caches
from pyramidkv import _mega_state_ref as _mega_ref
from pipeline.pyramidkv_config import PyramidKVPipelineConfig
from pipeline.logging_utils import BlockTimer, format_block_progress, get_log_mode, is_main_process, should_use_timestep_progress

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    """End-to-end inference pipeline for Self-Forcing + Pyramid Forcing.

    Owns the four collaborators required to turn a text prompt into a video:

      * ``self.generator`` — :class:`WanDiffusionWrapper` over the causal
        Wan2.1 transformer (loaded from a Self-Forcing DMD checkpoint).
      * ``self.text_encoder`` — UMT5-XXL text encoder producing the
        ``[B, 512, 4096]`` prompt embedding.
      * ``self.vae`` — 3D causal VAE with streaming decode.
      * ``self.kv_cache1[layer_idx]`` — per-layer ``AdaptiveKVCache`` (or
        :class:`MegaCache` when ``PYRAMIDKV_USE_MEGA_CACHE=1``) holding the
        per-head heterogeneous KV state.

    Driving an inference call (``inference()``) performs block-by-block
    autoregressive denoising under Self-Forcing's Double Pass: each block
    runs a noisy pass to get attention output, then a clean pass that
    commits the canonical K/V into the head caches before moving on. Frames
    are streamed to the VAE on overlapped CUDA streams so decoding hides
    behind the next block's denoising.

    All Pyramid-Forcing-specific behavior is controlled by the YAML ``args``: head
    classification CSV path, per-label capacity overrides, sink/middle/recent
    composition policies, and dynamic RoPE knobs.
    """

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

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.use_pyramidkv = getattr(args, "use_pyramidkv", False)
        self.pyramidkv_config = PyramidKVPipelineConfig.from_args(args, frame_seq_length=self.frame_seq_length)
        self.use_teacache = getattr(args, "use_teacache", False)
        if self.use_teacache:
            teacache_coefficients = getattr(args, "teacache_coefficients", None)
            if teacache_coefficients is not None:
                teacache_coefficients = [float(v) for v in teacache_coefficients]
            teacache_use_ret_coefficients = getattr(
                args,
                "teacache_use_ret_coefficients",
                getattr(args, "teacache_use_ret_steps", False),
            )
            self.generator.configure_teacache(
                rel_l1_thresh=float(getattr(args, "teacache_rel_l1_thresh", 0.08)),
                max_skip_steps=int(getattr(args, "teacache_max_skip_steps", 3)),
                coefficients=teacache_coefficients,
                variant=getattr(args, "teacache_model_variant", None),
                use_ret_steps=bool(teacache_use_ret_coefficients),
            )

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # torch.compile FFN layers for kernel fusion (8-16% speedup on FFN compute)
        self._compile_ffn = getattr(args, "compile_ffn", False)
        self._ffn_compiled = False

        # VAE decode mode:
        #   "streaming" — decode each block while DiT inference continues (overlap)
        #   "batch"     — collect latents, decode all after diffusion ends
        self.vae_decode_mode = getattr(args, "vae_decode_mode", "streaming")
        if self.vae_decode_mode not in ("streaming", "batch"):
            raise ValueError(
                f"vae_decode_mode must be 'streaming' or 'batch', got {self.vae_decode_mode!r}"
            )

    def _iter_adaptive_kv_caches(self):
        if self.kv_cache1 is None:
            return []
        return [cache for cache in self.kv_cache1 if isinstance(cache, AdaptiveKVCache)]

    def _set_adaptive_kv_profile(self, enabled: bool) -> None:
        for cache in self._iter_adaptive_kv_caches():
            cache.set_profile_enabled(enabled)
            if enabled:
                cache.reset_profile_stats()

    def _pop_adaptive_kv_profile_stats(self) -> dict[str, float]:
        totals = {
            "update_ms": 0.0,
            "collect_ms": 0.0,
            "pack_ms": 0.0,
            "rope_ms": 0.0,
            "cold_pack_count": 0.0,
            "refresh_pack_count": 0.0,
            "layout_reuse_count": 0.0,
            "layout_shape_changed_count": 0.0,
            "layout_anchor_changed_count": 0.0,
            "cuda_refresh_count": 0.0,
            "cuda_refresh_fallback_count": 0.0,
            "cuda_refresh_segment_count": 0.0,
            "cuda_refresh_total_len": 0.0,
            "readout_total_len": 0.0,
            "readout_max_seqlen": 0.0,
            "anchor_store_update_count": 0.0,
            "anchor_store_collect_count": 0.0,
            "anchor_store_fallback_count": 0.0,
            "anchor_store_anchor_count": 0.0,
            "anchor_store_token_count": 0.0,
            "cpp_strategy_update_count": 0.0,
            "cpp_strategy_collect_count": 0.0,
            "cpp_strategy_fallback_count": 0.0,
            "cpp_strategy_fallback_cpu_count": 0.0,
            "cpp_strategy_fallback_extension_count": 0.0,
            "cpp_strategy_fallback_policy_count": 0.0,
            "cpp_strategy_fallback_shape_count": 0.0,
            "cpp_strategy_fallback_inactive_count": 0.0,
            "cpp_strategy_anchor_count": 0.0,
            "cpp_strategy_token_count": 0.0,
            "cpp_strategy_materialize_count": 0.0,
            "cpp_strategy_materialize_token_count": 0.0,
            "rope_only_refresh_count": 0.0,
        }
        for cache in self._iter_adaptive_kv_caches():
            stats = cache.pop_profile_stats()
            for key in totals:
                totals[key] += float(stats.get(key, 0.0))
        return totals

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
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
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert batch_size == 1, (
            f"CausalInferencePipeline hardcodes batch_size=1 (VAE streaming requirement), got {batch_size}"
        )
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            pyramidkv_block_stats = []
            teacache_block_stats = []
            clean_pass_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            context_len = 0
            if self.use_pyramidkv and self.pyramidkv_config.pyramidkv_is_i2v and initial_latent is not None:
                context_len = self.pyramidkv_config.pyramidkv_context_len
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                context_len=context_len,
                max_frames=num_output_frames
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
                self.crossattn_cache[block_index]["prompt_v"] = None
            # reset kv cache
            # 根据实际类型区分 PyramidKV 模式和旧的 dict 模式，避免混用导致错误
            if self.kv_cache1 and hasattr(self.kv_cache1[0], "ctx") and hasattr(self.kv_cache1[0], "states_bytes_for_layer"):
                # MegaCache: re-build to wipe pool + per-head states.
                self.kv_cache1 = None
                self._initialize_kv_cache(
                    batch_size=batch_size,
                    dtype=noise.dtype,
                    device=noise.device,
                    context_len=0,
                    max_frames=num_output_frames,
                )
            elif self.kv_cache1 and isinstance(self.kv_cache1[0], PyramidKVCache):
                for cache in self.kv_cache1:
                    cache.reset()
            else:
                for block_index in range(len(self.kv_cache1)):
                    self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
        self._set_adaptive_kv_profile(profile)
        if self.use_teacache:
            self.generator.pop_teacache_stats()
        teacache_summary_enabled = (
            self.use_teacache
            and os.environ.get("PYRAMIDKV_TEACACHE_SUMMARY", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )

        # Initialize VAE streaming decode
        self.vae.decode_streaming_init(device=noise.device, dtype=noise.dtype)
        pixel_chunks = []
        latent_chunks: list[torch.Tensor] = []  # only used in batch mode
        vae_stream = torch.cuda.Stream()
        # GPU-side event chain: VAE stream waits on (a) previous VAE chunk (decoder is
        # stateful — causal conv buffer) and (b) main stream finishing denoised_pred.
        # Main stream NEVER waits on VAE — DiT continues enqueuing the next block.
        prev_vae_event: Optional[torch.cuda.Event] = None
        if profile:
            vae_chunk_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        def _submit_vae_chunk(latent: torch.Tensor) -> None:
            """Dispatch one VAE decode chunk. Streaming: enqueue on vae_stream with
            event-based sync. Batch: defer — just hold the latent.
            """
            if self.vae_decode_mode == "batch":
                latent_chunks.append(latent.detach().clone())
                return
            nonlocal prev_vae_event
            if profile:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
            if prev_vae_event is not None:
                vae_stream.wait_event(prev_vae_event)
            main_done = torch.cuda.Event()
            main_done.record()  # main stream progress
            vae_stream.wait_event(main_done)
            with torch.cuda.stream(vae_stream):
                if profile:
                    start_evt.record(vae_stream)
                pixel_chunks.append(self.vae.decode_streaming_chunk(latent))
                if profile:
                    end_evt.record(vae_stream)
            prev_vae_event = torch.cuda.Event()
            prev_vae_event.record(vae_stream)
            if profile:
                vae_chunk_events.append((start_evt, end_evt))

        # Step 2: Cache context feature
        # Compile only FFN layers. Cross-attention reuses crossattn_cache tensors
        # across model invocations, which is unsafe with Dynamo/CUDAGraphs.
        if self._compile_ffn and not self._ffn_compiled:
            # G3a: opt-in CUDA Graph mode via PYRAMIDKV_CUDA_GRAPH=1.
            # reduce-overhead enables Inductor's CUDAGraph wrapper; expected
            # ~5-10% util gain on dispatch-bound bs=1 FFN at the cost of
            # longer warmup and stricter shape stability.
            compile_mode = (
                "reduce-overhead"
                if os.environ.get("PYRAMIDKV_CUDA_GRAPH") == "1"
                else "max-autotune-no-cudagraphs"
            )
            for block in self.generator.model.blocks:
                block.ffn = torch.compile(block.ffn, mode=compile_mode)
            self._ffn_compiled = True

        current_start_frame = 0
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
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                _submit_vae_chunk(initial_latent[:, :1])
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                _submit_vae_chunk(current_ref_latents)
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        total_denoise_blocks = len(all_num_frames)

        # Pre-allocate reusable buffers for the denoising loop
        max_block_frames = max(all_num_frames)
        log_mode = get_log_mode()  # "tqdm" | "print" | "silent"
        timestep_buf = torch.empty(
            [batch_size, max_block_frames], device=noise.device, dtype=torch.int64)
        noise_buf = torch.empty(
            [batch_size * max_block_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)
        scalar_buf = torch.empty(
            [batch_size * max_block_frames], device=noise.device, dtype=torch.long)

        for block_index, current_num_frames in enumerate(all_num_frames, start=1):
            if profile:
                block_start.record()
                clean_pass_start = torch.cuda.Event(enable_timing=True)
                clean_pass_end = torch.cuda.Event(enable_timing=True)
                self._set_adaptive_kv_profile(True)

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Slice buffers for current block size
            ts_buf = timestep_buf[:, :current_num_frames]
            n_flat = batch_size * current_num_frames
            ns_buf = noise_buf[:n_flat]
            sc_buf = scalar_buf[:n_flat]
            block_label = format_block_progress(
                block_index=block_index,
                total_blocks=total_denoise_blocks,
                block_start_frame=current_start_frame + 1,
                total_frames=num_output_frames,
            )
            use_timestep_bar = log_mode == "tqdm"
            use_print_timer = log_mode == "print" and is_main_process()
            block_timer = BlockTimer(block_label, enabled=use_print_timer)
            block_timer.__enter__()

            # Step 3.1: Spatial denoising loop
            # The bar counts ``len(denoising_step_list) + 1`` because each
            # block runs N denoise forwards followed by one clean-pass forward
            # (Step 3.3 below) — the clean pass IS the KV-cache write, so it's
            # legitimately part of the block's compute. Bar updated once per
            # forward; final tick fires after the clean pass.
            self._timestep_pbar = None
            if self.use_teacache:
                self.generator.reset_teacache_state()
            if use_timestep_bar:
                self._timestep_pbar = tqdm(
                    total=len(self.denoising_step_list) + 1,
                    desc=block_label,
                    leave=True,
                    dynamic_ncols=True,
                )
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep (reuse pre-allocated buffer)
                timestep = ts_buf.fill_(current_timestep)
                teacache_force_calc = (
                    index == 0 or index == len(self.denoising_step_list) - 1
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_update_mode="noisy",
                        teacache_force_calc=teacache_force_calc,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    ns_buf.normal_()
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        ns_buf,
                        sc_buf.fill_(next_timestep),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_update_mode="noisy",
                        teacache_force_calc=teacache_force_calc,
                    )

                if self._timestep_pbar is not None:
                    self._timestep_pbar.update(1)

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            if profile:
                clean_pass_start.record()
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                cache_update_mode="clean",
                skip_x0=True,
            )
            if profile:
                clean_pass_end.record()
            if self._timestep_pbar is not None:
                # Mark the clean-pass forward as the final tick so the bar
                # ends at total = len(denoising_step_list) + 1 instead of
                # showing 4/5 (which would look like the block is incomplete).
                self._timestep_pbar.set_postfix_str("clean", refresh=False)
                self._timestep_pbar.update(1)
                self._timestep_pbar.close()
                self._timestep_pbar = None

            # Step 3.4: VAE decode — streaming (overlaps with next block's DiT)
            # or deferred to after the diffusion loop (batch mode).
            _submit_vae_chunk(denoised_pred)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)
                clean_pass_times.append(clean_pass_start.elapsed_time(clean_pass_end))
                pyramidkv_block_stats.append(self._pop_adaptive_kv_profile_stats())
                teacache_block_stats.append(self.generator.pop_teacache_stats())

            # Step 3.5: update the start and end frame indices
            current_start_frame += current_num_frames
            block_timer.__exit__(None, None, None)

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)

        # Step 4: Synchronize VAE stream and concatenate decoded chunks
        if self.vae_decode_mode == "batch":
            # Batch mode: decode all latent chunks now (diffusion is done)
            for chunk in latent_chunks:
                pixel_chunks.append(self.vae.decode_streaming_chunk(chunk))
        else:
            # Streaming mode: wait for the last VAE chunk to finish
            vae_stream.synchronize()
        video = torch.cat(pixel_chunks, dim=1)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.vae.decode_streaming_finalize()

        if profile:
            # Compute VAE timing from recorded events (requires sync already done above)
            if self.vae_decode_mode == "streaming" and vae_chunk_events:
                torch.cuda.synchronize()
                vae_time = sum(
                    s.elapsed_time(e) for s, e in vae_chunk_events
                )
            else:
                vae_time = 0.0
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            pyramidkv_totals = {
                key: sum(stats[key] for stats in pyramidkv_block_stats)
                for key in ("update_ms", "collect_ms", "pack_ms", "rope_ms")
            }
            if pyramidkv_block_stats:
                print(
                    "  - PyramidKV breakdown:"
                    f" update={pyramidkv_totals['update_ms']:.2f} ms,"
                    f" collect={pyramidkv_totals['collect_ms']:.2f} ms,"
                    f" pack={pyramidkv_totals['pack_ms']:.2f} ms,"
                    f" rope={pyramidkv_totals['rope_ms']:.2f} ms"
                )
                if os.environ.get("PYRAMIDKV_LAYOUT_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on"):
                    layout_totals = {
                        key: sum(float(stats.get(key, 0.0)) for stats in pyramidkv_block_stats)
                        for key in (
                            "cold_pack_count",
                            "refresh_pack_count",
                            "layout_reuse_count",
                            "layout_shape_changed_count",
                            "layout_anchor_changed_count",
                            "cuda_refresh_count",
                            "cuda_refresh_fallback_count",
                            "cuda_refresh_segment_count",
                            "cuda_refresh_total_len",
                            "anchor_store_update_count",
                            "anchor_store_collect_count",
                            "anchor_store_fallback_count",
                            "anchor_store_anchor_count",
                            "anchor_store_token_count",
                            "cpp_strategy_update_count",
                            "cpp_strategy_collect_count",
                            "cpp_strategy_fallback_count",
                            "cpp_strategy_fallback_cpu_count",
                            "cpp_strategy_fallback_extension_count",
                            "cpp_strategy_fallback_policy_count",
                            "cpp_strategy_fallback_shape_count",
                            "cpp_strategy_fallback_inactive_count",
                            "cpp_strategy_anchor_count",
                            "cpp_strategy_token_count",
                            "cpp_strategy_materialize_count",
                            "cpp_strategy_materialize_token_count",
                            "rope_only_refresh_count",
                        )
                    }
                    readout_total_len = max(float(stats.get("readout_total_len", 0.0)) for stats in pyramidkv_block_stats)
                    readout_max_seqlen = max(float(stats.get("readout_max_seqlen", 0.0)) for stats in pyramidkv_block_stats)
                    print(
                        "  - PyramidKV layout:"
                        f" cold_pack_count={layout_totals['cold_pack_count']:.0f},"
                        f" refresh_pack_count={layout_totals['refresh_pack_count']:.0f},"
                        f" layout_reuse_count={layout_totals['layout_reuse_count']:.0f},"
                        f" layout_shape_changed_count={layout_totals['layout_shape_changed_count']:.0f},"
                        f" layout_anchor_changed_count={layout_totals['layout_anchor_changed_count']:.0f},"
                        f" cuda_refresh_count={layout_totals['cuda_refresh_count']:.0f},"
                        f" cuda_refresh_fallback_count={layout_totals['cuda_refresh_fallback_count']:.0f},"
                        f" cuda_refresh_segment_count={layout_totals['cuda_refresh_segment_count']:.0f},"
                        f" cuda_refresh_total_len={layout_totals['cuda_refresh_total_len']:.0f},"
                        f" anchor_store_update_count={layout_totals['anchor_store_update_count']:.0f},"
                        f" anchor_store_collect_count={layout_totals['anchor_store_collect_count']:.0f},"
                        f" anchor_store_fallback_count={layout_totals['anchor_store_fallback_count']:.0f},"
                        f" anchor_store_anchor_count={layout_totals['anchor_store_anchor_count']:.0f},"
                        f" anchor_store_token_count={layout_totals['anchor_store_token_count']:.0f},"
                        f" cpp_strategy_update_count={layout_totals['cpp_strategy_update_count']:.0f},"
                        f" cpp_strategy_collect_count={layout_totals['cpp_strategy_collect_count']:.0f},"
                        f" cpp_strategy_fallback_count={layout_totals['cpp_strategy_fallback_count']:.0f},"
                        f" cpp_strategy_fallback_cpu_count={layout_totals['cpp_strategy_fallback_cpu_count']:.0f},"
                        f" cpp_strategy_fallback_extension_count={layout_totals['cpp_strategy_fallback_extension_count']:.0f},"
                        f" cpp_strategy_fallback_policy_count={layout_totals['cpp_strategy_fallback_policy_count']:.0f},"
                        f" cpp_strategy_fallback_shape_count={layout_totals['cpp_strategy_fallback_shape_count']:.0f},"
                        f" cpp_strategy_fallback_inactive_count={layout_totals['cpp_strategy_fallback_inactive_count']:.0f},"
                        f" cpp_strategy_anchor_count={layout_totals['cpp_strategy_anchor_count']:.0f},"
                        f" cpp_strategy_token_count={layout_totals['cpp_strategy_token_count']:.0f},"
                        f" cpp_strategy_materialize_count={layout_totals['cpp_strategy_materialize_count']:.0f},"
                        f" cpp_strategy_materialize_token_count={layout_totals['cpp_strategy_materialize_token_count']:.0f},"
                        f" readout_total_len={readout_total_len:.0f},"
                        f" readout_max_seqlen={readout_max_seqlen:.0f},"
                        f" rope_only_refresh_count={layout_totals['rope_only_refresh_count']:.0f}"
                    )
            if teacache_block_stats:
                full_calls = sum(stats["full_calls"] for stats in teacache_block_stats)
                skipped_calls = sum(stats["skipped_calls"] for stats in teacache_block_stats)
                print(
                    "  - TeaCache:"
                    f" full_calls={full_calls}, skipped_calls={skipped_calls}"
                )
            for i, block_time in enumerate(block_times, start=1):
                stats = pyramidkv_block_stats[i - 1] if i - 1 < len(pyramidkv_block_stats) else None
                tea_stats = teacache_block_stats[i - 1] if i - 1 < len(teacache_block_stats) else None
                clean_ms = clean_pass_times[i - 1] if i - 1 < len(clean_pass_times) else 0.0
                if stats is None:
                    print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
                    continue
                tea_text = ""
                if tea_stats is not None:
                    tea_text = (
                        f" | teacache(full={tea_stats['full_calls']},"
                        f" skip={tea_stats['skipped_calls']})"
                    )
                print(
                    f"    - Block {i} generation time: {block_time:.2f} ms "
                    f"({100 * block_time / diffusion_time:.2f}% of diffusion) | "
                    f"clean={clean_ms:.2f} ms | "
                    f"pyramidkv(update={stats['update_ms']:.2f}, collect={stats['collect_ms']:.2f}, "
                    f"pack={stats['pack_ms']:.2f}, rope={stats['rope_ms']:.2f})"
                    f"{tea_text}"
                )
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")
        elif teacache_summary_enabled:
            stats = self.generator.pop_teacache_stats()
            total_noisy_calls = stats["full_calls"] + stats["skipped_calls"]
            skip_rate = 0.0 if total_noisy_calls == 0 else 100.0 * stats["skipped_calls"] / total_noisy_calls
            print(
                "TeaCache summary:"
                f" full_calls={stats['full_calls']},"
                f" skipped_calls={stats['skipped_calls']},"
                f" skip_rate={skip_rate:.1f}%",
                flush=True,
            )

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, context_len=0, max_frames=None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if self.use_pyramidkv:
            hc = self.pyramidkv_config
            num_layers = self.generator.model.num_layers
            num_heads = self.generator.model.num_heads
            head_dim = self.generator.model.dim // num_heads
            if self.local_attn_size != -1:
                base_capacity_tokens = self.local_attn_size * self.frame_seq_length
            else:
                base_capacity_tokens = 32760
                if max_frames is not None:
                     base_capacity_tokens = max(base_capacity_tokens, max_frames * self.frame_seq_length)

            default_capacity = hc.pyramidkv_default_capacity or base_capacity_tokens
            config = PyramidKVConfig(
                hc.pyramidkv_config_path,
                num_layers=num_layers,
                num_heads=num_heads,
                default_capacity=default_capacity,
                strategy_reduction_factor=hc.pyramidkv_strategy_factor,
                code_map=hc.pyramidkv_code_map,
                head_type_csv_path=hc.pyramidkv_policy_csv_path,
                drop_heads_csv_path=hc.pyramidkv_drop_heads_csv_path,
                soft_ablate_heads_csv_path=hc.pyramidkv_soft_ablate_csv_path,
                af_policy_enabled=hc.pyramidkv_af_policy_enabled,
                af_csv_path=hc.pyramidkv_af_csv_path,
                af_group_dir=hc.pyramidkv_af_group_dir,
                af_manifest_path=hc.pyramidkv_af_manifest_path,
                frame_seq_length=hc.pyramidkv_frame_seq_length,
            )
            # Build compositions with strategy params from pipeline config
            if hc.use_adaptive_pyramidkv and hc.pyramidkv_policy_csv_path:
                compositions = build_compositions(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    capacities=config.capacity_map,
                    csv_path=hc.pyramidkv_policy_csv_path,
                    cyclic_enabled=hc.cyclic_enabled,
                    cyclic_period=hc.cyclic_period,
                    cyclic_bucket_cap=hc.cyclic_bucket_cap,
                    cyclic_dynamic_rope=hc.cyclic_dynamic_rope,
                    cyclic_osc_only=hc.cyclic_osc_only,
                    lag_enabled=hc.lag_enabled,
                    lag_offsets=hc.pyramidkv_lag_offsets,
                    lag_history=hc.pyramidkv_lag_history,
                    lag_dynamic_rope=hc.lag_dynamic_rope,
                    stride_enabled=hc.stride_enabled,
                    stride_interval=hc.stride_interval,
                    stride_capacity=hc.stride_capacity,
                    stride_dynamic_rope=hc.stride_dynamic_rope,
                    merge_enabled=hc.merge_enabled,
                    merge_patch_size=hc.merge_patch_size,
                    merge_capacity=hc.merge_capacity,
                    merge_dynamic_rope=hc.merge_dynamic_rope,
                    osc_sink_frames=hc.pyramidkv_osc_sink_frames,
                    stable_sink_frames=hc.pyramidkv_stable_sink_frames,
                    recent_frames=hc.pyramidkv_recent_frames,
                    stable_recent_frames=hc.pyramidkv_stable_recent_frames,
                    label_sink_frames_map=hc.pyramidkv_label_sink_frames_map,
                    label_recent_frames_map=hc.pyramidkv_label_recent_frames_map,
                    label_stride_enabled_map=hc.pyramidkv_label_stride_enabled_map,
                    label_stride_interval_map=hc.pyramidkv_label_stride_interval_map,
                    label_phase_bucket_map=hc.pyramidkv_label_phase_bucket_map,
                    label_lag_offsets_map=hc.pyramidkv_label_lag_offsets_map,
                    label_merge_enabled_map=hc.pyramidkv_label_merge_enabled_map,
                    label_merge_patch_size_map=hc.pyramidkv_label_merge_patch_size_map,
                    label_merge_capacity_map=hc.pyramidkv_label_merge_capacity_map,
                )
                config.compositions = compositions
                config.policies = compositions

            # Env-gated MegaCache path (Python sink/recent + C++ middle).
            # Set PYRAMIDKV_USE_MEGA_CACHE=1 to take the C++ kernel path.
            if (
                os.environ.get("PYRAMIDKV_USE_MEGA_CACHE") == "1"
                and hc.use_adaptive_pyramidkv
                and hc.pyramidkv_policy_csv_path
            ):
                osc_sink = int(hc.pyramidkv_osc_sink_frames or 1)
                stable_sink = int(hc.pyramidkv_stable_sink_frames or 3)
                stable_recent = int(hc.pyramidkv_stable_recent_frames or hc.pyramidkv_recent_frames)
                max_sink_frames = max(osc_sink, stable_sink)
                max_recent_frames = max(int(hc.pyramidkv_recent_frames), stable_recent)
                # Middle pool sized for stride/lag/cyclic worst-case occupancy.
                max_middle_frames = int(_mega_ref.MAX_T_KEYED)
                # Plan B — flat pack workspaces are sized for one attend
                # call's worst case (= num_frame_per_block chunks). Give 2×
                # headroom for safety / future num_frame_per_block bumps;
                # at L=30/H=12/MT=31/F=1560 this still saves ~6 GB vs the
                # old all-layer MTT sizing.
                max_attend_chunks = max(8, int(self.num_frame_per_block) * 2)
                self.kv_cache1 = build_mega_caches(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    frame_seqlen=int(self.frame_seq_length),
                    max_sink_frames=max_sink_frames,
                    max_middle_frames=max_middle_frames,
                    max_recent_frames=max_recent_frames,
                    max_attend_chunks=max_attend_chunks,
                    compositions=compositions,
                    device=str(device),
                    kv_dtype=("bfloat16" if dtype == torch.bfloat16 else "float16"),
                    sink_time_mapping_mode=hc.pyramidkv_dynamic_rope_mode,
                    sink_time_clamp_min=int(hc.sink_time_clamp_min),
                    sink_time_clamp_max=int(hc.sink_time_clamp_max),
                    decoupled_sink_time_lag=int(hc.decoupled_sink_time_lag),
                    history_time_mapping_mode=hc.history_time_mapping_mode,
                    history_relative_t_max=int(hc.history_relative_t_max),
                    history_time_soft_factor=float(hc.history_time_soft_factor),
                    freqs=self.generator.model.freqs,
                )
                return

            self.kv_cache1 = [
                (
                    AdaptiveKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.pyramidkv_is_i2v,
                        context_len=context_len,
                        sink_len=hc.pyramidkv_sink_tokens,
                        tail_len=hc.pyramidkv_dynamic_capacity,
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
                        sink_time_mapping_mode=hc.pyramidkv_dynamic_rope_mode,
                        sink_time_clamp_min=hc.sink_time_clamp_min,
                        sink_time_clamp_max=hc.sink_time_clamp_max,
                        history_time_mapping_mode=hc.history_time_mapping_mode,
                        history_relative_t_max=hc.history_relative_t_max,
                        history_time_soft_factor=hc.history_time_soft_factor,
                        use_osc_frame_mode=hc.cyclic_enabled,
                        phase_period=hc.cyclic_period,
                        phase_bucket_capacity_frames=hc.cyclic_bucket_cap,
                        local_tail_frames=hc.pyramidkv_recent_frames,
                        phase_sink_for_osc_only=hc.cyclic_osc_only,
                        phase_sink_dynamic_rope=hc.cyclic_dynamic_rope,
                        use_osc_lag_mode=hc.lag_enabled,
                        osc_lag_offsets_frames=hc.pyramidkv_lag_offsets,
                        osc_lag_history_frames=hc.pyramidkv_lag_history,
                        osc_lag_dynamic_rope=hc.lag_dynamic_rope,
                        disable_first_sink_for_osc_heads=hc.pyramidkv_disable_osc_sink,
                        use_stable_head_policies=hc.pyramidkv_stable_policy_enabled,
                        stable_sink_frames=hc.pyramidkv_stable_sink_frames,
                        osc_sink_frames=hc.pyramidkv_osc_sink_frames,
                        stable_recent_frames=hc.pyramidkv_stable_recent_frames,
                        use_af_head_policies=hc.pyramidkv_af_policy_enabled,
                        af_recent_frames_map=hc.pyramidkv_af_recent_frames_map,
                        af_phase_bucket_map=hc.pyramidkv_af_phase_bucket_map,
                        af_lag_offsets_map=hc.pyramidkv_af_lag_offsets_map,
                        af_sink_frames_map=hc.pyramidkv_af_sink_frames_map,
                        af_stride_enabled_map=hc.pyramidkv_af_stride_enabled_map,
                        label_recent_frames_map=hc.pyramidkv_label_recent_frames_map,
                        label_phase_bucket_map=hc.pyramidkv_label_phase_bucket_map,
                        label_lag_offsets_map=hc.pyramidkv_label_lag_offsets_map,
                        label_sink_frames_map=hc.pyramidkv_label_sink_frames_map,
                        label_stride_enabled_map=hc.pyramidkv_label_stride_enabled_map,
                        capture_frame_id_mode=hc.pyramidkv_capture_frame_id_mode,
                        readout_cache_enabled=hc.pyramidkv_readout_cache_enabled,
                        prompt_value_cache_enabled=hc.pyramidkv_prompt_v_cache_enabled,
                    )
                    if hc.use_adaptive_pyramidkv else
                    PyramidKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.pyramidkv_is_i2v,
                        context_len=context_len,
                        frame_seq_length=hc.pyramidkv_frame_seq_length,
                        prompt_value_cache_enabled=hc.pyramidkv_prompt_v_cache_enabled,
                    )
                )
                for layer_idx in range(num_layers)
            ]
            # Soft ablation controls are runtime knobs on cache objects.
            for cache in self.kv_cache1:
                cache.soft_ablate_region = str(hc.pyramidkv_soft_ablate_region)
                cache.soft_ablate_scale = float(hc.pyramidkv_soft_ablate_scale)
        else:
            kv_cache1 = []
            if self.local_attn_size != -1:
                # Use the local attention size to compute the KV cache size
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Use the default KV cache size or the size required for max_frames
                kv_cache_size = 32760
                if max_frames is not None:
                    kv_cache_size = max(kv_cache_size, max_frames * self.frame_seq_length)

            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })

            self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
                "prompt_v": None,
            })
        self.crossattn_cache = crossattn_cache
