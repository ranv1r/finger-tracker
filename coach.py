"""Touch-typing finger coach.

Watches your hands through the webcam, matches each keystroke to the fingertip
that made it, and (optionally) erases the current word when you used the wrong
finger.

    python coach.py                      # watch only, prints verdicts
    python coach.py --erase              # actually delete wrong-fingered words
    python coach.py --record data.csv    # log features for training a classifier

Press ESC in the video window to quit.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from pynput import keyboard

import attribution
import keymap
from attribution import MCPS, TIPS, Attributor, FingerPose, PoseFrame

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

WORD_BREAKS = {" ", "\t", "\n"}

DEFAULT_CALIBRATION = "calibration.json"


# ------------------------------------------------------------------ camera ---
def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera, warming up macOS/AVFoundation's device discovery first.

    Opening a non-zero camera index cold (before any camera has been opened
    this process) can silently fail to stream frames on macOS -- touching
    index 0 first primes the capture backend. See list_cameras.py, which
    works around the same issue by probing index 0 before anything else.
    """
    if index != 0:
        warm = cv2.VideoCapture(0)
        warm.read()
        warm.release()
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}")
    return cap


def pin_exposure(cap: cv2.VideoCapture, exposure: float) -> None:
    """Best-effort manual exposure lock. Property meanings/units are backend-
    and driver-specific -- if your camera ignores this, it's not a bug here."""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # "manual" on many UVC/AVFoundation drivers
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


def measure_fps(cap: cv2.VideoCapture, n: int = 30) -> float:
    """Read n frames and report the achieved rate. A dark scene forces long
    auto-exposure, which shows up here as a low, jittery rate well before it
    shows up as bad attribution accuracy."""
    t0 = time.perf_counter()
    got = 0
    for _ in range(n):
        ok, _ = cap.read()
        if ok:
            got += 1
    elapsed = time.perf_counter() - t0
    return got / elapsed if elapsed > 0 else 0.0


@dataclass
class KeyEvent:
    t: float
    char: str


# --------------------------------------------------------------- listener ----
class KeyLogger:
    """Global keystroke hook. Pushes timestamped events; swallows our own."""

    def __init__(self):
        self.events: deque[KeyEvent] = deque()
        self.word: list[str] = []
        self._suppress_backspaces = 0
        self._listener = keyboard.Listener(on_press=self._on_press)
        self.controller = keyboard.Controller()

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def _on_press(self, key):
        t = time.perf_counter()
        if key == keyboard.Key.backspace:
            if self._suppress_backspaces > 0:
                self._suppress_backspaces -= 1
                return
            if self.word:
                self.word.pop()
            return
        char = " " if key == keyboard.Key.space else getattr(key, "char", None)
        if not char:
            return
        if char in WORD_BREAKS:
            self.word.clear()
        else:
            self.word.append(char)
        self.events.append(KeyEvent(t, char))

    def erase_word(self):
        """Delete everything typed since the last word boundary."""
        n = len(self.word)
        if not n:
            return 0
        self._suppress_backspaces += n
        for _ in range(n):
            self.controller.tap(keyboard.Key.backspace)
            time.sleep(0.004)
        self.word.clear()
        return n


# ------------------------------------------------------------------ vision ---
class HandTracker:
    def __init__(self, model_path: str, flip_handedness: bool = False):
        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.flip = flip_handedness
        self.t0 = time.perf_counter()

    def pose(self, frame_bgr, t_capture: float) -> tuple[PoseFrame, object]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((t_capture - self.t0) * 1000)
        result = self.landmarker.detect_for_video(image, ts_ms)

        h, w = frame_bgr.shape[:2]
        pose = PoseFrame(t=t_capture, hand_scale=1.0)
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            label = handed[0].category_name  # "Left" / "Right"
            if self.flip:
                label = "Right" if label == "Left" else "Left"
            prefix = "L_" if label == "Left" else "R_"

            px = [(lm.x * w, lm.y * h) for lm in lms]
            scale = max(float(np.hypot(px[0][0] - px[9][0], px[0][1] - px[9][1])), 1e-6)
            pose.hand_scale = scale

            for name, tip in TIPS.items():
                mcp = MCPS[name]
                flexion = float(np.hypot(px[tip][0] - px[mcp][0],
                                         px[tip][1] - px[mcp][1])) / scale
                pose.fingers[prefix + name] = FingerPose(tip_px=px[tip], flexion=flexion)
        return pose, result


# -------------------------------------------------------------------- draw ---
def draw_overlay(frame, pose: PoseFrame, key_map: keymap.KeyMap, status: str, colour):
    for char, entry in key_map.entries.items():
        if entry.px is None:
            continue
        pitch = key_map.pitch_near(char)
        c = keymap.key_colour(entry, pitch)
        x, y = int(entry.px[0]), int(entry.px[1])
        cv2.circle(frame, (x, y), 2, c, -1)
    for name, fp in pose.fingers.items():
        x, y = int(fp.tip_px[0]), int(fp.tip_px[1])
        tone = (60, 200, 255) if name.startswith("L_") else (255, 160, 60)
        cv2.circle(frame, (x, y), 6, tone, -1)
        cv2.putText(frame, name[2:5], (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, tone, 1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(frame, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return frame


def _extra_margin_for(key_map: keymap.KeyMap, char: str) -> float:
    """Demand a wider score margin on keys whose calibrated position is
    shaky, so a noisy sample is more likely to abstain than mis-attribute."""
    entry = key_map.entries.get(keymap.normalize(char) or char)
    if entry is None or entry.absent or entry.spread_px is None:
        return 0.0
    pitch = key_map.pitch_near(char)
    spread_ratio = entry.spread_px / max(pitch, 1e-6)
    return min(0.4, 1.5 * spread_ratio)


# -------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hand_landmarker.task")
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--erase", action="store_true",
                    help="actually delete the word on a wrong-finger verdict")
    ap.add_argument("--flip-handedness", action="store_true",
                    help="use if left/right labels are swapped (non-mirrored camera)")
    ap.add_argument("--min-margin", type=float, default=0.35,
                    help="abstain when the top two fingers are closer than this")
    ap.add_argument("--record", help="append per-finger features to this CSV")
    ap.add_argument("--pin-exposure", type=float,
                    help="lock manual exposure to this value (backend-specific units) "
                         "to fight motion blur in a dark scene")
    args = ap.parse_args()

    try:
        key_map = keymap.KeyMap.load(args.calibration)
        print(f"loaded {args.calibration} ({len(key_map.entries)} keys)")
    except (FileNotFoundError, OSError):
        key_map = keymap.KeyMap()
        print(f"!! no calibration found at {args.calibration} -- every key will "
              "read as uncalibrated and verdicts will abstain. Run "
              "'python calibrate.py --type' first.")

    mismatch = key_map.check_camera(args.width, args.height)
    if mismatch:
        print(f"!! {mismatch}")

    tracker = HandTracker(args.model, args.flip_handedness)
    attributor = Attributor(min_margin=args.min_margin)
    logger = KeyLogger()
    logger.start()

    buffer: deque[PoseFrame] = deque(maxlen=120)   # ~4 s at 30 fps
    pending: list[KeyEvent] = []
    settle = attributor.lookahead + 0.02
    stats = {"ok": 0, "wrong": 0, "unsure": 0}
    status, colour = "typing...", (200, 200, 200)

    recorder = event_id = None
    if args.record:
        f = open(args.record, "a", newline="")
        recorder = csv.writer(f)
        if f.tell() == 0:
            recorder.writerow(["event_id", "char", "finger", "is_expected",
                               *attribution.FEATURE_NAMES])
        event_id = 0

    cap = open_camera(args.camera, args.width, args.height)
    if args.pin_exposure is not None:
        pin_exposure(cap, args.pin_exposure)

    fps = measure_fps(cap)
    print(f"measured capture rate: {fps:.1f} fps")
    if fps < 24:
        print(f"!! {fps:.1f} fps is low -- expect motion-blurred fingertips and "
              "jittery attribution. Add more light on your hands, or lock exposure "
              "with --pin-exposure (cv2.CAP_PROP_AUTO_EXPOSURE/CAP_PROP_EXPOSURE).")

    try:
        while True:
            ok, frame = cap.read()
            t_capture = time.perf_counter()
            if not ok:
                continue
            pose, _ = tracker.pose(frame, t_capture)
            buffer.append(pose)

            while logger.events:
                pending.append(logger.events.popleft())

            now = time.perf_counter()
            ready = [e for e in pending if now - e.t >= settle]
            pending = [e for e in pending if now - e.t < settle]

            for event in ready:
                extra_margin = _extra_margin_for(key_map, event.char)
                verdict = attributor.attribute(buffer, event.t, event.char,
                                               keymap_map=key_map, extra_margin=extra_margin)
                expected = keymap.expected_fingers(event.char)

                if recorder is not None and verdict.features:
                    for finger, feats in verdict.features.items():
                        recorder.writerow([event_id, event.char, finger,
                                           int(finger in expected),
                                           *[f"{feats[k]:.4f}"
                                             for k in attribution.FEATURE_NAMES]])
                    event_id += 1

                if verdict.finger is None:
                    stats["unsure"] += 1
                    status = f"'{event.char}' -> ? {verdict.reason}"
                    colour = (150, 150, 150)
                elif verdict.finger in expected:
                    stats["ok"] += 1
                    status = f"'{event.char}' -> {verdict.finger} OK"
                    colour = (80, 230, 120)
                else:
                    stats["wrong"] += 1
                    want = "/".join(sorted(expected))
                    status = f"'{event.char}' -> {verdict.finger}, want {want}"
                    colour = (70, 70, 255)
                    if args.erase:
                        n = logger.erase_word()
                        status += f"  [erased {n}]"

            total = sum(stats.values()) or 1
            hud = (f"{status}   |  ok {stats['ok']}  wrong {stats['wrong']}"
                   f"  unsure {stats['unsure']}  ({100*stats['ok']//total}%)")
            cv2.imshow("finger coach", draw_overlay(frame, pose, key_map, hud, colour))
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        logger.stop()
        cap.release()
        cv2.destroyAllWindows()
        if args.record:
            f.close()
        print(stats)


if __name__ == "__main__":
    main()
