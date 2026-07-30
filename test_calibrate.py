"""Headless test for calibrate.py's Stage-A harvesting (Harvester / --replay).

Builds a synthetic frames+events fixture -- the same idea as test_attr.py's
`build()`, but as a recorded session on disk -- and checks that the harvester
places each key near where its pressing finger actually was, with no camera,
no GUI, and no human at a keyboard.
"""
import json
import math
import tempfile
from pathlib import Path

from calibrate import run_replay_mode

# One deliberately-moving finger per key, resting elsewhere between presses.
# Offsets are generously sized (a real deliberate keystroke during calibration
# travels the fingertip from a relaxed rest position, not a subtle flick) so
# the motion signal clearly dominates the resting fingers' near-zero score.
HOME = {"R_index": "j", "R_middle": "k", "L_index": "f"}
TARGET_PX = {"j": (612.0, 344.0), "k": (700.0, 344.0), "f": (418.0, 344.0)}
REST_PX = {"R_index": (600.0, 480.0), "R_middle": (700.0, 480.0), "L_index": (418.0, 480.0)}


def _press_burst(actor: str, target: tuple[float, float], t0: float, dt: float = 0.02,
                  n: int = 10, hold: int = 3):
    """Frames of `actor` travelling from rest to `target`, then holding there
    briefly (a real fingertip dwells on the key for a few frames); the event
    fires during the hold, so the nearest-frame lookup lands exactly on target."""
    frames = []
    rest = REST_PX[actor]
    for i in range(n + hold):
        a = min(i / (n - 1), 1.0)
        t = t0 + i * dt
        fingers = {}
        for finger, home_px in REST_PX.items():
            if finger == actor:
                x = rest[0] + a * (target[0] - rest[0])
                y = rest[1] + a * (target[1] - rest[1])
                flex = 0.9 - 0.5 * a
            else:
                x, y = home_px
                flex = 0.9
            fingers[finger] = {"tip_px": [x, y], "flexion": flex}
        frames.append({"t": t, "hand_scale": 100.0, "fingers": fingers})
    event_t = t0 + n * dt   # first frame of the hold
    return frames, event_t


def build_fixture(samples_per_key: int = 4) -> dict:
    frames, events = [], []
    t = 0.0
    for char, target in TARGET_PX.items():
        actor = next(f for f, home in HOME.items() if home == char)
        for _ in range(samples_per_key):
            burst, event_t = _press_burst(actor, target, t)
            frames.extend(burst)
            events.append({"t": event_t, "char": char})
            t = burst[-1]["t"] + 0.3   # idle gap: lets settle() drain the previous event
    # trailing idle frames so the very last pending sample still gets drained
    for i in range(6):
        tt = t + i * 0.02
        fingers = {f: {"tip_px": list(px), "flexion": 0.9} for f, px in REST_PX.items()}
        frames.append({"t": tt, "hand_scale": 100.0, "fingers": fingers})
    return {"camera": {"width": 1280, "height": 720}, "frames": frames, "events": events}


def test_harvester_places_keys_from_typing():
    fixture = build_fixture()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "fixture.json"
        path.write_text(json.dumps(fixture))
        km = run_replay_mode(str(path), order=list(TARGET_PX.keys()))

    for char, target in TARGET_PX.items():
        assert km.is_calibrated(char), f"'{char}' was never placed"
        px = km.pixel_of(char)
        err = math.hypot(px[0] - target[0], px[1] - target[1])
        assert err < 5.0, f"'{char}' placed at {px}, expected near {target} (err={err:.1f}px)"
        entry = km.entries[char]
        assert entry.source == "typed"
        assert entry.samples == 4
        assert entry.spread_px is not None and entry.spread_px < 5.0


if __name__ == "__main__":
    test_harvester_places_keys_from_typing()
    print("test_calibrate.py: harvester replay test passed")
