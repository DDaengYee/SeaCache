#!/usr/bin/env python3
"""Synthetic sanity tests for the FLUX cache gate."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "FLUX"))

from cache_gate import decide_cache_refresh  # noqa: E402


def simulate(distances, gate, delta_acc, delta_single=None):
    acc = 0.0
    reasons = []
    refreshes = []
    for distance in distances:
        refresh, acc, reason, single_hit, acc_hit = decide_cache_refresh(
            gate=gate,
            distance=distance,
            acc_before=acc,
            delta_acc=delta_acc,
            delta_single=delta_single,
        )
        reasons.append(reason)
        refreshes.append(refresh)
    return reasons, refreshes


def test_local_spike_guard():
    distances = [0.1, 0.1, 0.6, 0.1]
    acc_reasons, acc_refreshes = simulate(distances, "accumulated", 1.0, 0.5)
    dual_reasons, dual_refreshes = simulate(distances, "dual", 1.0, 0.5)

    assert acc_reasons == ["skip", "skip", "skip", "skip"]
    assert acc_refreshes == [False, False, False, False]
    assert dual_reasons == ["skip", "skip", "single", "skip"]
    assert dual_refreshes == [False, False, True, False]


def test_accumulated_refresh_without_single_hit():
    distances = [0.2, 0.2, 0.2, 0.2, 0.25]
    reasons, refreshes = simulate(distances, "dual", 1.0, 0.5)

    assert "single" not in reasons
    assert reasons == ["skip", "skip", "skip", "skip", "acc"]
    assert refreshes == [False, False, False, False, True]


def main():
    test_local_spike_guard()
    test_accumulated_refresh_without_single_hit()
    print("cache gate sanity tests passed")


if __name__ == "__main__":
    main()
