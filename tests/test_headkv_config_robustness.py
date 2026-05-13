import os
import tempfile

from headkv.config import HeadKVConfig


def _tmp_csv(lines):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        for line in lines:
            f.write(line + "\n")
        path = f.name
    return path


def test_unknown_classification_labels_fallback_to_default_capacity():
    # Classification format: row length > 3, values are class labels (not capacities).
    path = _tmp_csv(
        [
            "-1,1,2,3",
            "1,1,-1,2",
        ]
    )
    try:
        cfg = HeadKVConfig(
            config_path=path,
            num_layers=2,
            num_heads=4,
            default_capacity=100,
            code_map={"-1": 40, "1": 100},
        )
        layer0 = cfg.get_layer_capacities(0)
        layer1 = cfg.get_layer_capacities(1)
        # Unknown labels "2","3" should NOT be interpreted as capacities 2/3.
        assert layer0 == [40, 100, 100, 100]
        assert layer1 == [100, 100, 40, 100]
    finally:
        os.unlink(path)


def test_explicit_capacity_csv_still_supported():
    # Explicit format: layer_idx, head_idx, capacity
    path = _tmp_csv(
        [
            "0,0,12",
            "0,1,34",
            "1,0,56",
        ]
    )
    try:
        cfg = HeadKVConfig(
            config_path=path,
            num_layers=2,
            num_heads=2,
            default_capacity=99,
        )
        assert cfg.get_layer_capacities(0) == [12, 34]
        assert cfg.get_layer_capacities(1) == [56, 99]
    finally:
        os.unlink(path)
