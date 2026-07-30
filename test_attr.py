"""Synthetic test: does attribution pick the moving finger?"""
import math
from attribution import Attributor, FingerPose, PoseFrame
import keymap

HOME = {"R_index": "j", "R_middle": "k", "R_ring": "l", "R_pinky": ";",
        "L_index": "f", "L_middle": "d", "L_ring": "s", "L_pinky": "a"}

PX_SCALE = 40.0


def build_keymap() -> keymap.KeyMap:
    """A synthetic KeyMap in pixel space, standing in for a real calibration."""
    km = keymap.KeyMap()
    for ch, (x, y) in keymap.ANSI_REFERENCE_XY.items():
        km.set_pixel(ch, (x * PX_SCALE, y * PX_SCALE), source="typed", samples=4)
    return km


KM = build_keymap()


def build(target_key, actor, n=10, dt=1/30):
    """Frames where `actor` travels from its home key to target_key by frame n-1."""
    frames = []
    tgt = keymap.ANSI_REFERENCE_XY[target_key]
    for i in range(n):
        a = i / (n - 1)
        pf = PoseFrame(t=i * dt, hand_scale=100.0)
        for finger, home in HOME.items():
            hx, hy = keymap.ANSI_REFERENCE_XY[home]
            if finger == actor:
                x, y = hx + a * (tgt[0] - hx), hy + a * (tgt[1] - hy)
                flex = 0.9 - 0.15 * a          # curls as it strikes
            else:
                x, y = hx, hy                   # resting, with a little jitter
                x += 0.01 * math.sin(i)
                flex = 0.9
            pf.fingers[finger] = FingerPose(tip_px=(x * PX_SCALE, y * PX_SCALE), flexion=flex)
        frames.append(pf)
    return frames

att = Attributor()
cases = [("j", "R_index"), ("u", "R_index"), ("k", "R_middle"),
         ("l", "R_ring"), ("k", "R_index"), ("a", "L_pinky"), ("d", "L_middle")]
for key, actor in cases:
    buf = build(key, actor)
    t_event = buf[-1].t - 0.02
    v = att.attribute(buf, t_event, key, keymap_map=KM)
    exp = sorted(keymap.expected_fingers(key))
    hit = "correct" if v.finger == actor else "MISS"
    print(f"key {key!r:4} actually pressed by {actor:9} -> verdict {str(v.finger):9} "
          f"margin {v.margin:6.2f}  [{hit}] expected-by-map {exp} {v.reason}")
