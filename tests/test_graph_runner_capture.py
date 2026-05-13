"""M2.2 test: GraphedDiffusionRunner capture + replay round-trip.

Captures a small synthetic forward (linear → relu → linear) into noisy +
clean graphs, marks steady, then replays under varying inputs and
verifies the output matches the eager forward bit-exactly.

Mirrors the contract M2.2 needs against the real diffusion block forward:
- capture_noisy / capture_clean record graphs against a shared mempool
- step() replays the right slot per pass_kind
- input is copy_'d into the static slot tensor (caller-supplied size must
  match the capture-time shape)
- output is cloned from the static buffer
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
class TestGraphRunnerCapture:
    def _make_forward(self, device):
        torch.manual_seed(0)
        w0 = torch.randn(8, 8, device=device, dtype=torch.float32)
        w1 = torch.randn(8, 4, device=device, dtype=torch.float32)
        # Per pass_kind branching to verify clean uses a different captured graph.
        bias = {0: torch.zeros(4, device=device, dtype=torch.float32),
                1: torch.full((4,), 0.5, device=device, dtype=torch.float32)}

        def fwd(x: torch.Tensor, t: torch.Tensor, pass_kind: int) -> torch.Tensor:
            h = torch.relu(x @ w0 + t)
            return h @ w1 + bias[pass_kind]
        return fwd

    def test_capture_replay_matches_eager(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from pipeline.graph_runner import GraphedDiffusionRunner

        device = "cuda:0"
        fwd = self._make_forward(device)
        runner = GraphedDiffusionRunner(forward_fn=fwd, warmup_steps=3)

        x0 = torch.randn(2, 8, device=device, dtype=torch.float32)
        t0 = torch.tensor(0.1, device=device, dtype=torch.float32)

        # Required pre-capture warmup on side stream.
        runner.warmup_side_stream(x0, t0, 0)

        # Eager fallback before mark_steady.
        eager_n = fwd(x0, t0, 0)
        eager_c = fwd(x0, t0, 1)
        torch.testing.assert_close(runner.step(x0, t0, 0), eager_n)
        torch.testing.assert_close(runner.step(x0, t0, 1), eager_c)

        runner.capture_noisy(x0, t0)
        runner.capture_clean(x0, t0)
        runner.mark_steady()
        assert runner.steady

        # Replay under different inputs — output shape was fixed at capture
        # so we feed same-shape inputs and check bit-equivalent results.
        for _ in range(4):
            x = torch.randn(2, 8, device=device, dtype=torch.float32)
            t = torch.tensor(torch.rand(()).item(), device=device, dtype=torch.float32)
            torch.testing.assert_close(runner.step(x, t, 0), fwd(x, t, 0))
            torch.testing.assert_close(runner.step(x, t, 1), fwd(x, t, 1))

    def test_mark_steady_requires_both_graphs(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from pipeline.graph_runner import GraphedDiffusionRunner

        runner = GraphedDiffusionRunner(forward_fn=lambda x, t, k: x, warmup_steps=1)
        with pytest.raises(RuntimeError, match="noisy.*clean"):
            runner.mark_steady()

    def test_capture_slot_keyed_multi_timestep(self):
        """Multiple keyed slots — mirrors block forward with N denoising steps."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from pipeline.graph_runner import GraphedDiffusionRunner

        device = "cuda:0"
        fwd = self._make_forward(device)
        runner = GraphedDiffusionRunner(forward_fn=fwd, warmup_steps=2)

        x = torch.randn(2, 8, device=device, dtype=torch.float32)
        # Per-timestep capture: 3 noisy steps + 1 clean
        t_values = [torch.tensor(v, device=device, dtype=torch.float32) for v in (1.0, 0.75, 0.5, 0.25)]
        runner.warmup_side_stream(x, t_values[0], 0)
        for idx, t in enumerate(t_values[:-1]):
            runner.capture_slot((idx, "noisy"), x, t, pass_kind=0)
        runner.capture_slot((3, "clean"), x, t_values[-1], pass_kind=1)
        # Backward-compat shims still required for mark_steady
        runner.capture_noisy(x, t_values[0])
        runner.capture_clean(x, t_values[-1])
        runner.mark_steady()

        for idx in range(3):
            x_in = torch.randn(2, 8, device=device, dtype=torch.float32)
            torch.testing.assert_close(
                runner.step(x_in, t_values[idx], pass_kind=0, key=(idx, "noisy")),
                fwd(x_in, t_values[idx], 0),
            )
        x_in = torch.randn(2, 8, device=device, dtype=torch.float32)
        torch.testing.assert_close(
            runner.step(x_in, t_values[-1], pass_kind=1, key=(3, "clean")),
            fwd(x_in, t_values[-1], 1),
        )

    def test_replay_output_does_not_alias_static_buffer(self):
        """Returned output must be a clone — next replay overwrites the buffer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from pipeline.graph_runner import GraphedDiffusionRunner

        device = "cuda:0"
        fwd = self._make_forward(device)
        runner = GraphedDiffusionRunner(forward_fn=fwd, warmup_steps=2)
        x = torch.randn(2, 8, device=device, dtype=torch.float32)
        t = torch.tensor(0.0, device=device, dtype=torch.float32)
        runner.warmup_side_stream(x, t, 0)
        runner.capture_noisy(x, t)
        runner.capture_clean(x, t)
        runner.mark_steady()

        x1 = torch.randn(2, 8, device=device, dtype=torch.float32)
        out_a = runner.step(x1, t, 0)
        x2 = torch.randn(2, 8, device=device, dtype=torch.float32)
        out_b = runner.step(x2, t, 0)
        # out_a should still hold its value despite the second replay.
        torch.testing.assert_close(out_a, fwd(x1, t, 0))
        torch.testing.assert_close(out_b, fwd(x2, t, 0))
        assert not torch.equal(out_a, out_b)
