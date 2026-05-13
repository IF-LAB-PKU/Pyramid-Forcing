// M1 — Unified AnchorStore PerHeadState contract.
//
// Replaces the 5 Python strategy classes (cyclic / stride / lag / recent /
// merge) with one device-side struct array. mega_anchor_update_kernel and
// mega_anchor_collect_kernel dispatch on ``kind`` to apply the right state
// machine without per-head Python iteration.
//
// Storage relationship to existing pools:
//   - sink/middle/recent_pool : owned by PyramidKVCacheManager (M0)
//   - merge_*_pool / merge_accum_*: NEW in M1 (added to manager)
//   - PerHeadState[L, H]      : NEW in M1, ~256 byte/head, ~92 KB total
//
// Slot encoding convention:
//   - cyclic_slot[i] / tkey_slot[i] / merge_completed_slot[i] = -1 → empty
//   - all other slot fields are 0-based indices into the corresponding pool
//   - times stored as raw frame indices (current_t // frame_seqlen domain)
//
// Capacity rationale (pyramid-forcing G6 config):
//   period=6, bucket_cap=3 → cyclic_slot has 18 entries (6 phases × 3)
//   stride interval=6, capacity=4 → tkey_slot uses 4 of 18 entries
//   lag history_frames=21 → tkey_slot uses 18 (capped)
//   merge capacity=6, block_frames=4 → merge_completed[6] + merge_active[2]
//
// All struct fields are POD; safe to push from host once at strategy compile
// time. Update/collect kernels mutate in place (atomicCAS not needed since
// each (layer, head) maps to exactly one CTA).
#pragma once

#include <cstdint>

namespace adahead {

// ---------------------------------------------------------------------------
// Compile-time bounds. These must accommodate every supported config; bumping
// them is cheap (struct grows by O(KB) total). If a strategy needs more, fail
// the runtime assertion in PyramidKVCacheManager::configure_per_head() rather
// than overflowing.
// ---------------------------------------------------------------------------
constexpr int kMaxPhase           = 6;   // cyclic period upper bound
constexpr int kMaxBucket          = 4;   // cyclic bucket_cap upper bound (yaml: 4)
constexpr int kMaxTKeyed          = 24;  // shared by stride/lag history & middle pool sizing — must be ≥ kMaxPhase*kMaxBucket so cyclic_slot fits
constexpr int kMaxLagOffsets      = 4;   // distinct lag offsets per head
constexpr int kMaxMergeBlocks     = 6;   // finalized merge anchors retained
constexpr int kMaxMergeActive     = 2;   // simultaneously accumulating blocks
constexpr int kMaxMergeBlockFrames = 4;  // merge.block_frames = patch_size**2

enum StrategyKind : int8_t {
    SK_RECENT = 0,
    SK_CYCLIC = 1,
    SK_STRIDE = 2,
    SK_LAG    = 3,
    SK_MERGE  = 4,
};

// ---------------------------------------------------------------------------
// Per (layer, head) state.
//
// Field layout grouped by strategy_kind use; only fields whose kind matches
// are read in update/collect. Unused fields stay at sentinel values (-1) and
// don't cost runtime work — the dispatch is a single switch on ``kind``.
// ---------------------------------------------------------------------------
struct PerHeadState {
    // ===== shared metadata =====
    int8_t  kind;            // StrategyKind
    int8_t  sink_capacity;   // per-head sink frame budget (M5 step 5)
    int8_t  _pad0[2];

    int32_t period;          // cyclic
    int32_t bucket_cap;      // cyclic
    int32_t interval;        // stride
    int32_t capacity;        // stride / merge / lag history
    int32_t patch_size;      // merge — block_frames = patch_size * patch_size
    int32_t block_frames;    // merge — derived from patch_size, cached

    // lag offsets: collect time = current_t - lag_offsets[i]
    int32_t lag_offsets[kMaxLagOffsets];
    int32_t lag_offset_count;

    // ===== Cyclic state =====
    // Per phase: bucket_cap slots forming a ring. cyclic_cursor[phase] is the
    // next-write index modulo bucket_cap.
    //
    // cyclic_slot[phase * kMaxBucket + cursor] holds the middle_pool slot id;
    // cyclic_t[same_idx] holds the absolute frame t value (current_t domain).
    int32_t cyclic_slot[kMaxPhase * kMaxBucket]; // -1 = empty
    int32_t cyclic_t   [kMaxPhase * kMaxBucket]; // valid only if slot != -1
    int8_t  cyclic_cursor[kMaxPhase];

    // ===== Stride / Lag state (shared FIFO of t-keyed entries) =====
    // For stride: only entries where t % interval == 0 are stored
    // For lag:    every frame stored, capped at history_frames=capacity
    // Both: FIFO eviction when count == capacity
    //
    // tkey_slot[i] is the middle_pool slot id; tkey_t[i] is the t value
    // associated with that slot (so collect can do an O(N) scan to find the
    // requested t = current_t - offset).
    int32_t tkey_slot[kMaxTKeyed];   // -1 = empty
    int32_t tkey_t   [kMaxTKeyed];
    int8_t  tkey_count;              // saturates at capacity
    int8_t  tkey_head;               // next-write slot (ring buffer, wraps % capacity)
    int8_t  _pad1[2];

    // ===== Merge state =====
    // Finalized blocks: merge_completed_slot[i] points into merge_k_pool /
    // merge_v_pool / merge_pos_pool at slot i. merge_completed_count tracks
    // how many of MAX are populated (FIFO when full).
    int32_t merge_completed_slot     [kMaxMergeBlocks]; // -1 = empty
    int32_t merge_completed_block_id [kMaxMergeBlocks];
    int32_t merge_completed_start_t  [kMaxMergeBlocks];
    int32_t merge_completed_end_t    [kMaxMergeBlocks];
    int32_t merge_completed_median_t [kMaxMergeBlocks];
    int8_t  merge_completed_count;
    int8_t  _pad2[3];

    // Accumulating blocks: each slot holds intermediate sum_k / sum_v / count
    // state for a block whose 4 frames haven't all arrived yet. When the 4th
    // frame lands, the accumulator divides by tokens_per_group, copies into
    // merge_*_pool[completed_slot], and clears.
    //
    // merge_active_seen tracks which of the 4 frames have been seen; used to
    // detect duplicates (matching the Python ValueError check at merge.py:108).
    int32_t merge_active_block_id      [kMaxMergeActive]; // -1 = inactive
    int32_t merge_active_slot          [kMaxMergeActive]; // index into merge_accum_*
    int8_t  merge_active_seen          [kMaxMergeActive][kMaxMergeBlockFrames];
    int8_t  merge_active_complete_count[kMaxMergeActive];
    int8_t  _pad3[2];

    // ===== Cached per-strategy invariants (avoids per-call .item() syncs) =====
    // For merge: num_groups = (patch_h_count) * (patch_w_count). Filled on
    // first update via patch_x_count input from host, never re-read.
    int32_t cached_num_groups;
};

// Static size budget — keep room for growth without runtime alloc.
// After bumping kMaxBucket=4 / kMaxTKeyed=24 (cyclic_slot OOB fix), the
// struct grew to ~600 bytes. 30 layers × 12 heads × 1024 = 360 KB
// device-side, still negligible at H200 scale.
static_assert(sizeof(PerHeadState) <= 1024,
              "PerHeadState should stay compact (≤1024 bytes)");

// ---------------------------------------------------------------------------
// Pool layout constants (mirror Python config; M1 ships these as compile-time
// macros, M2 will plumb through PlanInfo).
// ---------------------------------------------------------------------------
constexpr int kMaxSinkFrames   = 3;
constexpr int kMaxMiddleFrames = 18;
constexpr int kMaxRecentFrames = 4;
constexpr int kMaxMergeTokens  = 420;  // patch_size=2 + 60×7 grid (worst case)

// Total tokens per head in flat output (sink + middle + recent only;
// merge tokens are accumulated separately and concatenated at pack time).
constexpr int kMaxTotalFrames = kMaxSinkFrames + kMaxMiddleFrames + kMaxRecentFrames;

}  // namespace adahead
