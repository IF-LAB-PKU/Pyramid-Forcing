"""Tests for headkv/config.py — HeadKVConfig A-F CSV loading."""
from __future__ import annotations

import csv
import os
import tempfile

import pytest

from headkv.config import HeadKVConfig


def _write_csv(path: str, rows: list[list[str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


class TestAfCsvLoadsAndConverts:
    def test_af_csv_loads_and_converts_case(self, tmp_path):
        csv_path = str(tmp_path / "af.csv")
        # 2 layers × 3 heads, lowercase input
        _write_csv(csv_path, [
            ["a", "b", "c"],
            ["d", "e", "f"],
        ])
        cfg = HeadKVConfig(
            config_path=None, num_layers=2, num_heads=3,
            af_policy_enabled=True, af_csv_path=csv_path,
        )
        assert cfg.af_group_map[0] == ["A", "B", "C"]
        assert cfg.af_group_map[1] == ["D", "E", "F"]

    def test_af_csv_handles_mixed_case(self, tmp_path):
        csv_path = str(tmp_path / "af.csv")
        _write_csv(csv_path, [["A", "b", "C"]])
        cfg = HeadKVConfig(
            config_path=None, num_layers=1, num_heads=3,
            af_policy_enabled=True, af_csv_path=csv_path,
        )
        assert cfg.af_group_map[0] == ["A", "B", "C"]

    def test_af_csv_skips_invalid_letters(self, tmp_path):
        csv_path = str(tmp_path / "af.csv")
        _write_csv(csv_path, [["a", "x", "b"]])
        cfg = HeadKVConfig(
            config_path=None, num_layers=1, num_heads=3,
            af_policy_enabled=True, af_csv_path=csv_path,
        )
        assert cfg.af_group_map[0][0] == "A"
        assert cfg.af_group_map[0][1] == ""   # "x" is invalid
        assert cfg.af_group_map[0][2] == "B"

    def test_af_csv_truncates_extra_cols(self, tmp_path):
        csv_path = str(tmp_path / "af.csv")
        _write_csv(csv_path, [["a", "b", "c", "d", "e"]])
        cfg = HeadKVConfig(
            config_path=None, num_layers=1, num_heads=3,
            af_policy_enabled=True, af_csv_path=csv_path,
        )
        # Only first 3 columns read
        assert cfg.af_group_map[0] == ["A", "B", "C"]

    def test_af_csv_pads_missing_layers(self, tmp_path):
        csv_path = str(tmp_path / "af.csv")
        _write_csv(csv_path, [["a", "b"]])
        cfg = HeadKVConfig(
            config_path=None, num_layers=3, num_heads=2,
            af_policy_enabled=True, af_csv_path=csv_path,
        )
        assert cfg.af_group_map[0] == ["A", "B"]
        # Missing layers default to empty
        assert cfg.af_group_map[1] == ["", ""]
        assert cfg.af_group_map[2] == ["", ""]

    def test_af_csv_priority_over_manifest(self, tmp_path):
        """When af_csv_path is provided, it takes priority over manifest."""
        csv_path = str(tmp_path / "af.csv")
        _write_csv(csv_path, [["a", "b"]])
        manifest_path = str(tmp_path / "manifest.csv")
        _write_csv(manifest_path, [["consensus_class_id", "file_name"]])
        cfg = HeadKVConfig(
            config_path=None, num_layers=1, num_heads=2,
            af_policy_enabled=True,
            af_csv_path=csv_path,
            af_manifest_path=manifest_path,
        )
        # CSV path should be used, not manifest
        assert cfg.af_group_map[0] == ["A", "B"]

    def test_af_disabled_when_flag_false(self, tmp_path):
        csv_path = str(tmp_path / "af.csv")
        _write_csv(csv_path, [["a", "b"]])
        cfg = HeadKVConfig(
            config_path=None, num_layers=1, num_heads=2,
            af_policy_enabled=False, af_csv_path=csv_path,
        )
        # Should remain default empty
        assert cfg.af_group_map[0] == ["", ""]
