import torch
import os
import csv
from typing import List, Optional


# Tracks config_paths already announced via the "Loading PyramidKV config" line.
# MegaCache rebuilds PyramidKVConfig on every prompt (the manager is wiped per
# inference call); printing once is helpful, printing 128× per run is just
# noise that mangles the progress bar.
_LOADED_CONFIG_PATHS: set = set()


class PyramidKVConfig:
    """Central configuration for PyramidKV caches across a whole model.

    Aggregates the per-(layer, head) metadata that cache instances need to
    decide their behavior:

      * **capacity_map** — token capacity per head, derived from the per-head
        classification CSV via ``code_map`` (e.g. ``{"-1": 1024, "1": 32768}``
        sets oscillating heads to a smaller window).
      * **label_map** — per-head class label loaded from the classification
        CSV (-1 = fluct/osc, 1 = compact/stable, 2 = sparse).
      * **drop_head_mask** / **soft_ablate_head_mask** — optional ablation
        masks loaded from CSVs.
      * **af_group_map** — optional 6-class A–F taxonomy assignment used by
        the fine-grained per-group strategy router.
      * **compositions** — populated externally by
        ``pyramidkv.factory.build_compositions``; the list-of-lists of
        :class:`HeadComposition` actually executed by adaptive caches.

    The class itself owns no GPU buffers — it is a pure metadata holder that
    individual ``PyramidKVCache`` / ``AdaptiveKVCache`` instances read at
    construction time.

    Args:
        config_path: Path to the classification CSV (30×12 matrix of labels).
            Pass ``None`` to keep the default uniform-capacity setup.
        num_layers: Total transformer layers in the model.
        num_heads: Attention heads per layer.
        default_capacity: Fallback capacity when a head's label is missing
            from ``code_map``.
        strategy_reduction_factor: When ``code_map`` is None, oscillating
            heads (label -1) get ``default_capacity // strategy_reduction_factor``.
        code_map: Mapping ``{label_str: capacity_int}`` overriding the
            reduction-factor heuristic.
        head_type_csv_path: Optional override for the classification CSV.
        drop_heads_csv_path: Optional CSV listing ``(layer, head)`` pairs
            whose self-attention output is zeroed.
        soft_ablate_heads_csv_path: Optional CSV listing ``(layer, head)``
            pairs subject to region-wise attention scaling.
        af_policy_enabled: Activate the A–F taxonomy router.
        af_csv_path / af_group_dir / af_manifest_path: A–F CSV and the
            per-group strategy directory.
        frame_seq_length: Tokens per frame (used by per-frame anchor logic).
    """

    def __init__(
        self,
        config_path: Optional[str],
        num_layers: int,
        num_heads: int,
        default_capacity: int = 32768,
        strategy_reduction_factor: int = 3,
        code_map: Optional[dict] = None,
        head_type_csv_path: Optional[str] = None,
        drop_heads_csv_path: Optional[str] = None,
        soft_ablate_heads_csv_path: Optional[str] = None,
        af_policy_enabled: bool = False,
        af_csv_path: Optional[str] = None,
        af_group_dir: Optional[str] = None,
        af_manifest_path: Optional[str] = None,
        frame_seq_length: Optional[int] = None,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.capacity_map = torch.full((num_layers, num_heads), default_capacity, dtype=torch.int32)
        # Keep original per-head class labels when classification CSV is used.
        # Default label "1" means non-oscillating.
        self.label_map = torch.full((num_layers, num_heads), 1, dtype=torch.int32)
        raw_code_map = code_map or {
            "-1": max(1, default_capacity // max(1, strategy_reduction_factor)),
            "1": default_capacity,
        }
        # Normalize keys to string to avoid int/str mismatch from YAML parsing.
        self.code_map = {str(k): int(v) for k, v in raw_code_map.items()}
        # 可选的 head 类型配置（默认为 configs/head_configs/best_labels.csv）
        self.head_type_csv_path = head_type_csv_path
        # 可选的 head drop 配置（CSV 中的 layer,head 会在 self-attn 输出侧被置零）
        self.drop_heads_csv_path = drop_heads_csv_path
        # 可选的 head soft-ablation 配置（CSV 中的 layer,head 会进行区域性注意力缩放）
        self.soft_ablate_heads_csv_path = soft_ablate_heads_csv_path
        # 可选的 A-F taxonomy 分组（按 layer/head 读取）
        self.af_policy_enabled = bool(af_policy_enabled)
        self.af_csv_path = af_csv_path
        self.af_group_dir = af_group_dir
        self.af_manifest_path = af_manifest_path
        # 每帧对应的 token 数（用于按帧进行 KV 选择），由 pipeline 传入
        self.frame_seq_length = frame_seq_length
        self.drop_head_mask = torch.zeros((num_layers, num_heads), dtype=torch.bool)
        self.soft_ablate_head_mask = torch.zeros((num_layers, num_heads), dtype=torch.bool)
        self.af_group_map = [["" for _ in range(num_heads)] for _ in range(num_layers)]

        if config_path and os.path.exists(config_path):
            if config_path not in _LOADED_CONFIG_PATHS:
                print(f"Loading PyramidKV config from {config_path}")
                _LOADED_CONFIG_PATHS.add(config_path)
            with open(config_path, 'r') as f:
                reader = csv.reader(f)
                classification_rows = []
                has_classification = False
                unknown_labels = set()
                for row in reader:
                    if not row:
                        continue
                    try:
                        if len(row) > 3:
                            has_classification = True
                            classification_rows.append([str(x).strip() for x in row if str(x).strip() != ""])
                            continue
                        # Format: layer_idx, head_idx, capacity
                        l, h, c = int(row[0]), int(row[1]), int(row[2])
                        if 0 <= l < num_layers and 0 <= h < num_heads:
                            self.capacity_map[l, h] = c
                    except (ValueError, IndexError):
                        continue
                if has_classification:
                    if len(classification_rows) != num_layers:
                        print(
                            f"Warning: classification rows ({len(classification_rows)}) != num_layers ({num_layers}); "
                            "missing layers will use default capacity."
                        )
                    for layer_idx in range(num_layers):
                        if layer_idx < len(classification_rows):
                            row = classification_rows[layer_idx]
                        else:
                            row = []
                        if row and len(row) != num_heads:
                            print(
                                f"Warning: layer {layer_idx} has {len(row)} heads, expected {num_heads}; "
                                "missing heads will use default capacity."
                            )
                        for head_idx in range(num_heads):
                            raw = str(row[head_idx]).strip() if head_idx < len(row) else "1"
                            try:
                                self.label_map[layer_idx, head_idx] = int(raw)
                            except ValueError:
                                self.label_map[layer_idx, head_idx] = 1
                            if raw in self.code_map:
                                self.capacity_map[layer_idx, head_idx] = int(self.code_map[raw])
                            else:
                                # In classification mode, unknown labels fall back to default capacity.
                                # This avoids accidentally treating class-id (e.g., "2") as tiny capacity 2.
                                self.capacity_map[layer_idx, head_idx] = default_capacity
                                unknown_labels.add(raw)
                    if unknown_labels:
                        print(
                            "Warning: unknown classification labels "
                            f"{sorted(unknown_labels)} not found in pyramidkv_code_map; "
                            f"falling back to default_capacity={default_capacity}."
                        )
        else:
            if config_path:
                print(f"Warning: PyramidKV config path {config_path} not found, using default capacity.")

        # Build per-head composition instances from a head-type CSV.
        self.compositions: list | None = None
        csv_path = self.head_type_csv_path
        if not csv_path:
            default_csv = os.path.join("configs", "head_configs", "best_labels.csv")
            if os.path.exists(default_csv):
                csv_path = default_csv

        if csv_path and os.path.exists(csv_path):
            try:
                from .factory import build_compositions

                self.compositions = build_compositions(
                    num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    capacities=self.capacity_map,
                    csv_path=csv_path,
                )
            except Exception as e:
                print(f"Warning: failed to build head compositions from {csv_path}: {e}")
                self.compositions = None

        # Legacy alias for backward compatibility
        self.policies = self.compositions

        # Optional: load explicit drop-head list (layer,head).
        drop_csv = self.drop_heads_csv_path
        if drop_csv:
            if os.path.exists(drop_csv):
                loaded = 0
                with open(drop_csv, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or len(row) < 2:
                            continue
                        try:
                            layer_idx = int(str(row[0]).strip())
                            head_idx = int(str(row[1]).strip())
                        except ValueError:
                            # Skip header/non-numeric rows.
                            continue
                        if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                            if not self.drop_head_mask[layer_idx, head_idx]:
                                loaded += 1
                            self.drop_head_mask[layer_idx, head_idx] = True
                print(f"Loading PyramidKV drop-head list from {drop_csv} (num_heads={loaded})")
            else:
                print(f"Warning: PyramidKV drop-head path {drop_csv} not found, no heads dropped.")

        # Optional: load explicit soft-ablation head list (layer,head).
        soft_csv = self.soft_ablate_heads_csv_path
        if soft_csv:
            if os.path.exists(soft_csv):
                loaded = 0
                with open(soft_csv, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row or len(row) < 2:
                            continue
                        try:
                            layer_idx = int(str(row[0]).strip())
                            head_idx = int(str(row[1]).strip())
                        except ValueError:
                            continue
                        if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                            if not self.soft_ablate_head_mask[layer_idx, head_idx]:
                                loaded += 1
                            self.soft_ablate_head_mask[layer_idx, head_idx] = True
                print(f"Loading PyramidKV soft-ablate list from {soft_csv} (num_heads={loaded})")
            else:
                print(f"Warning: PyramidKV soft-ablate path {soft_csv} not found, soft-ablation disabled.")

        # Optional: load A-F taxonomy groups (layer,head -> group-id in {A..F}).
        if self.af_policy_enabled:
            valid_groups = {"A", "B", "C", "D", "E", "F"}

            # Priority 1: matrix CSV (30×12, values a-f)
            if self.af_csv_path and os.path.exists(self.af_csv_path):
                loaded_total = 0
                loaded_by_group = {g: 0 for g in valid_groups}
                with open(self.af_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for layer_idx, row in enumerate(reader):
                        if layer_idx >= num_layers:
                            break
                        for head_idx, val in enumerate(row):
                            if head_idx >= num_heads:
                                break
                            group = str(val).strip().upper()
                            if group in valid_groups:
                                self.af_group_map[layer_idx][head_idx] = group
                                loaded_total += 1
                                loaded_by_group[group] += 1
                if loaded_total > 0:
                    print(
                        f"Loading PyramidKV A-F groups from matrix CSV {self.af_csv_path} "
                        f"(total={loaded_total}, A={loaded_by_group['A']}, B={loaded_by_group['B']}, "
                        f"C={loaded_by_group['C']}, D={loaded_by_group['D']}, E={loaded_by_group['E']}, "
                        f"F={loaded_by_group['F']})"
                    )
                else:
                    print(f"Warning: A-F matrix CSV {self.af_csv_path} loaded but no valid entries found.")

            # Priority 2: manifest + per-group CSV files (legacy)
            else:
                class_to_group = {
                    "A_RECENT_TRACKER": "A",
                    "B_RHYTHM_DRIVER": "B",
                    "C_WINDOW_SMOOTHER": "C",
                    "D_CYCLE_ATTENDER": "D",
                    "E_ANCHOR_SPARSE": "E",
                    "F_BROAD_LOCAL": "F",
                }
                group_dir = self.af_group_dir
                manifest = self.af_manifest_path
                if not group_dir and manifest:
                    group_dir = os.path.dirname(manifest)
                if not manifest and group_dir:
                    cand = os.path.join(group_dir, "taxonomy_heads_manifest.csv")
                    if os.path.exists(cand):
                        manifest = cand

                entries: list[tuple[str, str]] = []
                if manifest and os.path.exists(manifest):
                    with open(manifest, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            class_id = str(row.get("consensus_class_id", "")).strip()
                            file_name = str(row.get("file_name", "")).strip()
                            if class_id and file_name:
                                entries.append((class_id, file_name))
                elif group_dir and os.path.isdir(group_dir):
                    for file_name in os.listdir(group_dir):
                        if not file_name.endswith("_heads.csv"):
                            continue
                        class_id = file_name[: -len("_heads.csv")]
                        entries.append((class_id, file_name))
                else:
                    print(
                        "Warning: A-F policy is enabled but taxonomy_heads source is missing. "
                        f"group_dir={group_dir}, manifest={manifest}"
                    )

                loaded_total = 0
                loaded_by_group = {g: 0 for g in valid_groups}
                resolved_dir = group_dir if group_dir else ""
                for class_id, file_name in entries:
                    group = class_to_group.get(class_id)
                    if not group:
                        continue
                    file_path = file_name if os.path.isabs(file_name) else os.path.join(resolved_dir, file_name)
                    if not os.path.exists(file_path):
                        continue
                    with open(file_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                layer_idx = int(str(row.get("layer", "")).strip())
                                head_idx = int(str(row.get("head", "")).strip())
                            except ValueError:
                                continue
                            if 0 <= layer_idx < num_layers and 0 <= head_idx < num_heads:
                                if self.af_group_map[layer_idx][head_idx] != group:
                                    loaded_total += 1
                                    loaded_by_group[group] += 1
                                self.af_group_map[layer_idx][head_idx] = group

                if loaded_total > 0:
                    print(
                        "Loading PyramidKV A-F groups "
                        f"(total={loaded_total}, A={loaded_by_group['A']}, B={loaded_by_group['B']}, "
                        f"C={loaded_by_group['C']}, D={loaded_by_group['D']}, E={loaded_by_group['E']}, "
                        f"F={loaded_by_group['F']})"
                    )
                else:
                    print("Warning: A-F policy enabled but no valid layer/head entries were loaded.")

    def get_layer_capacities(self, layer_idx: int) -> List[int]:
        return self.capacity_map[layer_idx].tolist()

    def get_layer_labels(self, layer_idx: int) -> List[int]:
        return self.label_map[layer_idx].tolist()

    def get_layer_drop_mask(self, layer_idx: int) -> List[bool]:
        return self.drop_head_mask[layer_idx].tolist()

    def get_layer_soft_ablate_mask(self, layer_idx: int) -> List[bool]:
        return self.soft_ablate_head_mask[layer_idx].tolist()

    def get_layer_af_groups(self, layer_idx: int) -> List[str]:
        return list(self.af_group_map[layer_idx])
