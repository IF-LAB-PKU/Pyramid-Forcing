#!/usr/bin/env python
"""One-click self-check for Pyramid Forcing's dynamic RoPE (``window_clamp``).

It answers the question raised in issue #1: *"does the first block remap the
3 frames' RoPE straight to the far end of the training window (t≈21) instead
of staying continuous?"*

How it works
------------
The C++/CUDA "MegaCache" planner ``mega_plan_multi_cuda`` returns, per cache
segment, the field ``anchor_t_remap`` — this is the *exact* time coordinate
that is fed into the fused RoPE pack kernel (``_ops.pyramidkv_pack``). This
script hooks that function, records ``(sync_t_raw, sink_t_remap)`` for layer 0
across every autoregressive block, then prints a PASS/FAIL summary.

What it proves
--------------
* **Block 0** (sync_t = 0): sink ``t_remap == 0`` → positions stay continuous,
  NOT pushed to the far end of the window.
* **sync_t > 21**: ``t_remap = sync_t - 21`` so the query-key *relative
  distance* is pinned to the 21-frame training window — exactly what
  ``map_sink_time(..., "window_clamp", 18, 21)`` is designed to do.

Usage (identical CLI to ``inference.py``)
-----------------------------------------
    CUDA_VISIBLE_DEVICES=0 PYRAMIDKV_LOG_MODE=print \\
    uv run --no-sync python tools/verify_rope_window_clamp.py \\
        --config_path configs/pyramid-forcing.yaml \\
        --checkpoint_path checkpoints/self_forcing_dmd.pt \\
        --data_path prompts/MovieGenVideoBench_num32.txt \\
        --output_folder /tmp/rope_check --use_ema --num_output_frames 72

A 72-frame run (24 blocks) is enough to cross the t=21 boundary and watch the
clamp engage. The generated video under ``--output_folder`` is a throwaway.
"""
import os
import sys

# This probe hooks the C++/CUDA "MegaCache" planner, so make sure that path is
# active. Bare ``inference.py`` only takes it when PYRAMIDKV_USE_MEGA_CACHE=1
# (scripts/run_pyramid_forcing.sh sets this for you; the README quick-start
# does not). setdefault respects an explicit override.
os.environ.setdefault("PYRAMIDKV_USE_MEGA_CACHE", "1")

import pyramidkv._mega_state_ops as _mso

# num_training_frames for Wan2.1-T2V-1.3B / Self-Forcing.
TRAIN_WINDOW = 21

_records = []  # (sync_t_raw, sink_t_remap) for layer 0, every planner call
_orig_plan = _mso.mega_plan_multi_cuda
_state = {"max_sync_t": -1}


class _EnoughData(BaseException):
    """Raised once the first prompt's blocks have all been observed, so the
    probe stops instead of grinding through the whole prompt list. Subclasses
    BaseException (not Exception) so a stray ``except Exception`` in the
    pipeline can't swallow the early-stop signal."""


def _arg(kwargs, args, name, pos):
    """Read a planner arg whether it arrived as a keyword or positionally.

    Signature: mega_plan_multi_cuda(mgr, states_bytes, layer_idx,
    current_t_list, pass_kind, ...) → layer_idx is pos 2, current_t_list pos 3.
    """
    if name in kwargs:
        return kwargs[name]
    return args[pos] if len(args) > pos else None


_DEBUG = bool(int(os.environ.get("ROPE_VERIFY_DEBUG", "0")))
_dbg_left = [3]


def _hooked_plan(*args, **kwargs):
    out = _orig_plan(*args, **kwargs)
    if _DEBUG and _dbg_left[0] > 0:
        sys.stderr.write(f"[verify-debug] nargs={len(args)} kwargs={sorted(kwargs)}\n")
        sys.stderr.flush()
        _dbg_left[0] -= 1
    layer_idx = _arg(kwargs, args, "layer_idx", 2)
    if layer_idx != 0:
        return out
    try:
        # out = (cu_seqlens_k, src_kind, src_slot_global, seg_lengths,
        #        dst_offsets, anchor_t_raw, anchor_t_remap)
        src_kind = out[1].detach().to("cpu").reshape(-1)
        t_remap = out[6].detach().to("cpu").reshape(-1)
        cur_t = _arg(kwargs, args, "current_t_list", 3)
        sync_t = int(list(cur_t)[0]) if hasattr(cur_t, "__iter__") else int(cur_t)
        # sync_t resetting to 0 after we've advanced = a new prompt started;
        # one prompt already covers every sync_t we need, so stop here.
        if sync_t == 0 and _state["max_sync_t"] > 0:
            raise _EnoughData
        _state["max_sync_t"] = max(_state["max_sync_t"], sync_t)
        # first sink segment (src_kind == 0) belongs to chunk-0 / head-0
        sink_idx = (src_kind == 0).nonzero(as_tuple=True)[0]
        if len(sink_idx):
            _records.append((sync_t, int(t_remap[sink_idx[0]])))
    except _EnoughData:
        raise
    except Exception:  # never let the probe break a real inference run
        pass
    return out


_mso.mega_plan_multi_cuda = _hooked_plan

# Forward every CLI arg straight to inference.py and run it in-process so the
# monkey-patch above stays live. ``exec`` (rather than runpy) is used on
# purpose: it keeps the already-patched ``pyramidkv`` modules on the call
# path. Run from the repository root so ``inference.py`` resolves. One prompt
# is enough; the hook aborts the run as soon as the second prompt would start.
sys.argv = ["inference.py"] + sys.argv[1:]
_infer = os.path.join(os.getcwd(), "inference.py")
if not os.path.exists(_infer):
    sys.exit(f"inference.py not found in CWD ({os.getcwd()}); run from the repo root.")
try:
    with open(_infer) as _f:
        _code = compile(_f.read(), "inference.py", "exec")
    exec(_code, {"__name__": "__main__", "__file__": "inference.py"})
except (SystemExit, _EnoughData):
    pass

# ---------------------------------------------------------------- summary ----
# Keep the first plan per block (t_remap depends only on sync_t + mode, not on
# which denoise pass produced it).
by_block = {}
for sync_t, remap in _records:
    by_block.setdefault(sync_t, remap)

rows = sorted(by_block.items())
ok = bool(rows)
print("\n" + "=" * 58)
print("  Pyramid Forcing — dynamic RoPE (window_clamp) self-check")
print("=" * 58)
print(f"{'block sync_t':>13} | {'sink t_remap':>12} | {'Q-K rel dist':>12}")
print("-" * 44)
for sync_t, remap in rows:
    rel = sync_t - remap
    flag = ""
    if sync_t == 0 and remap != 0:
        ok, flag = False, "  <-- unexpected"
    if sync_t > TRAIN_WINDOW and rel != TRAIN_WINDOW:
        ok, flag = False, "  <-- unexpected"
    print(f"{sync_t:>13} | {remap:>12} | {rel:>12}{flag}")
print("-" * 44)
b0 = by_block.get(0)
if b0 is not None:
    verdict = "continuous (== 0), NOT mapped to far end (21)" if b0 == 0 else "UNEXPECTED"
    print(f"[block 0 ] sink t_remap = {b0}  ->  {verdict}")
print(f"[clamp   ] for sync_t > {TRAIN_WINDOW}: Q-K relative distance pinned to {TRAIN_WINDOW}")
print("RESULT   :", "PASS  window_clamp behaves as designed"
      if ok else "FAIL  unexpected RoPE mapping")
print("=" * 58)
