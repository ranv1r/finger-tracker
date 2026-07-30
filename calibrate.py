"""Keyboard calibration: measure where each key actually is, in camera pixels.

Two ways to build a calibration.json, and one to fix it:

    python calibrate.py --type              bootstrap by typing (recommended first run)
    python calibrate.py --edit              hand-place / fix individual keys
    python calibrate.py --replay session.json --out calibration.json
                                             headless: replay a recorded frames+events
                                             fixture instead of the camera (for testing)

Why not click 4 keys and fit a homography (the old approach)? A planar
transform assumes the keyboard is flat. Curved, split, or otherwise
nonstandard boards aren't, and the error is structural -- it can't be fixed by
clicking more carefully. So calibration here measures each key's position
directly instead of assuming a layout:

  --type  the primary path. When you press a key, the pressing fingertip *is*
           that key's location, by definition. No geometric assumptions about
           the board are needed at all.
  --edit  the fallback / refinement path. Load whatever calibration exists (or
           start blank) over a frozen frame and correct it by hand -- select a
           key by clicking or typing its character, move it, or fix a whole
           row at once.

Everything is stored in image pixels (see keymap.KeyMap) -- attribution.py
never sees a keyboard coordinate space.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from pynput import keyboard

import attribution
import keymap
from attribution import Attributor, FingerPose, PoseFrame
from coach import HandTracker, measure_fps, open_camera, pin_exposure

SAMPLES_PER_KEY = 4
REJECT_MARGIN_MULT = 2.5      # outlier-reject threshold, in local key pitches
FALLBACK_REJECT_PX = 150.0    # used before enough keys exist to measure a real pitch
MIN_KEYS_FOR_PITCH_REJECT = 4

HIT_RADIUS = 12               # px, for click-to-select in --edit

# Extended key codes for arrow keys, as returned by cv2.waitKeyEx() -- these
# vary by OpenCV GUI backend, so we check several known sets. All well above
# the 0-255 ASCII range, so they can't collide with plain letter/number keys.
ARROW_UP = {2490368, 63232, 65362}
ARROW_DOWN = {2621440, 63233, 65364}
ARROW_LEFT = {2424832, 63234, 65361}
ARROW_RIGHT = {2555904, 63235, 65363}


# ==================================================================== A. ===
# Stage A: bootstrap by typing.


class TypeCapture:
    """Global key listener for --type mode.

    Forwards printable characters as calibration samples; Enter skips the
    current key (marks it absent), Esc quits. Kept separate from coach.py's
    KeyLogger because calibration needs raw Enter/Esc control events, not
    word-erase bookkeeping.
    """

    def __init__(self):
        self.queue: deque[tuple[str, float]] = deque()
        self.control: deque[str] = deque()
        self._listener = keyboard.Listener(on_press=self._on_press)

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def _on_press(self, key):
        t = time.perf_counter()
        if key == keyboard.Key.esc:
            self.control.append("quit")
            return
        if key == keyboard.Key.enter:
            self.control.append("skip")
            return
        char = " " if key == keyboard.Key.space else getattr(key, "char", None)
        if char:
            self.queue.append((char, t))


class Harvester:
    """Drives Stage-A calibration: prompts one key at a time, scores each
    keystroke on motion + flex only (positions are what we're measuring, so
    proximity can't be used yet), and accumulates a median position per key.

    Fed incrementally via on_frame/on_key/on_control, so the same logic runs
    live (camera + TypeCapture) and in tests (a prerecorded frame/event list,
    via --replay) -- see test_calibrate.py.
    """

    def __init__(self, order, protected: keymap.KeyMap | None = None,
                 camera_meta: dict | None = None,
                 samples_per_key: int = SAMPLES_PER_KEY):
        self.km = keymap.KeyMap(camera=camera_meta)
        if protected is not None:
            for k, e in protected.entries.items():
                if e.source == "edited":
                    self.km.entries[k] = e
        self.order = [c for c in order if not self._is_protected(c)]
        self.samples_per_key = samples_per_key
        self.attributor = Attributor(use_position_features=False, min_margin=0.3, min_frames=3)
        self.buffer: deque[PoseFrame] = deque(maxlen=90)
        self._pending: list[tuple[str, float]] = []
        self._settle = self.attributor.lookahead + 0.02
        self._samples: dict[str, list[tuple[float, float]]] = {}
        self.idx = 0
        self.status = ""
        self.done = len(self.order) == 0
        self._now = 0.0

    def _is_protected(self, char: str) -> bool:
        e = self.km.entries.get(char)
        return e is not None and e.source == "edited"

    @property
    def current_key(self) -> str | None:
        if self.done or self.idx >= len(self.order):
            return None
        return self.order[self.idx]

    @property
    def progress(self) -> tuple[int, int, int, int]:
        char = self.current_key
        have = len(self._samples.get(char, [])) if char else self.samples_per_key
        return have, self.samples_per_key, self.idx, len(self.order)

    def on_frame(self, pose: PoseFrame) -> None:
        self.buffer.append(pose)
        self._now = pose.t
        self._drain_pending()

    def on_key(self, char: str, t: float) -> None:
        if self.done:
            return
        if keymap.normalize(char) == self.current_key:
            self._pending.append((self.current_key, t))

    def on_control(self, cmd: str) -> None:
        if cmd == "skip" and not self.done:
            char = self.current_key
            self.km.set_absent(char)
            self._samples.pop(char, None)
            self.status = f"skipped '{char}'"
            self._advance()

    def _advance(self) -> None:
        self.idx += 1
        if self.idx >= len(self.order):
            self.done = True

    def _reject_threshold(self, char: str) -> float:
        if len(self.km.entries) >= MIN_KEYS_FOR_PITCH_REJECT:
            return REJECT_MARGIN_MULT * self.km.pitch_near(char)
        return FALLBACK_REJECT_PX

    def _drain_pending(self) -> None:
        if self.done:
            return
        ready = [p for p in self._pending if self._now - p[1] >= self._settle]
        self._pending = [p for p in self._pending if self._now - p[1] < self._settle]
        for char, t_evt in ready:
            if self.done or char != self.current_key:
                continue
            verdict = self.attributor.attribute(self.buffer, t_evt, char)
            if verdict.finger is None:
                self.status = f"unclear ({verdict.reason}) -- try '{char}' again"
                continue
            frame_at = min(self.buffer, key=lambda f: abs(f.t - t_evt))
            fp = frame_at.fingers.get(verdict.finger)
            if fp is None:
                self.status = f"lost track of {verdict.finger} -- try '{char}' again"
                continue
            pos = fp.tip_px
            existing = self._samples.setdefault(char, [])
            if existing:
                mx = statistics.median(p[0] for p in existing)
                my = statistics.median(p[1] for p in existing)
                if math.hypot(pos[0] - mx, pos[1] - my) > self._reject_threshold(char):
                    self.status = f"'{char}' sample too far off the others -- try again"
                    continue
            existing.append(pos)
            self.status = f"'{char}' {len(existing)}/{self.samples_per_key} ({verdict.finger})"
            if len(existing) >= self.samples_per_key:
                xs = [p[0] for p in existing]
                ys = [p[1] for p in existing]
                mx, my = statistics.median(xs), statistics.median(ys)
                spread = statistics.median(math.hypot(x - mx, y - my) for x, y in existing)
                self.km.set_pixel(char, (mx, my), source="typed",
                                   samples=len(existing), spread_px=spread)
                self._advance()


def _pose_from_dict(d: dict) -> PoseFrame:
    pf = PoseFrame(t=d["t"], hand_scale=d.get("hand_scale", 1.0))
    for name, fd in d.get("fingers", {}).items():
        pf.fingers[name] = FingerPose(tip_px=tuple(fd["tip_px"]), flexion=fd.get("flexion", 0.0))
    return pf


def run_replay_mode(path: str, order=None, out: str | None = None) -> keymap.KeyMap:
    """Headless harvesting from a prerecorded fixture -- no camera, no GUI, no
    human at a keyboard. See test_calibrate.py."""
    with open(path) as f:
        data = json.load(f)
    frames = [_pose_from_dict(d) for d in data["frames"]]
    events = sorted(data.get("events", []), key=lambda e: e["t"])
    controls = sorted(data.get("controls", []), key=lambda c: c["t"])
    harvester = Harvester(order or list(keymap.TYPING_ORDER), camera_meta=data.get("camera"))

    ei = ci = 0
    for pf in frames:
        while ci < len(controls) and controls[ci]["t"] <= pf.t:
            harvester.on_control(controls[ci]["cmd"])
            ci += 1
        while ei < len(events) and events[ei]["t"] <= pf.t:
            harvester.on_key(events[ei]["char"], events[ei]["t"])
            ei += 1
        harvester.on_frame(pf)
        if harvester.done:
            break

    if out:
        harvester.km.save(out)
    where = "done" if harvester.done else f"stopped at {harvester.current_key!r}"
    print(f"replay: {len(harvester.km.entries)} keys placed ({where})")
    return harvester.km


def _draw_type_overlay(frame, harvester: Harvester, pose: PoseFrame) -> None:
    km = harvester.km
    for char, entry in km.entries.items():
        if entry.px is None:
            continue
        colour = keymap.key_colour(entry, km.pitch_near(char))
        x, y = int(entry.px[0]), int(entry.px[1])
        cv2.circle(frame, (x, y), 5, colour, -1)
        cv2.putText(frame, char, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    for name, fp in pose.fingers.items():
        x, y = int(fp.tip_px[0]), int(fp.tip_px[1])
        tone = (60, 200, 255) if name.startswith("L_") else (255, 160, 60)
        cv2.circle(frame, (x, y), 4, tone, 1)
        cv2.putText(frame, name[2:5], (x + 6, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tone, 1)

    have, need, idx, total = harvester.progress
    char = harvester.current_key
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (20, 20, 20), -1)
    prompt = (f"press  '{char}'  ({have}/{need})   [{idx}/{total} keys]   "
              f"Enter=skip key  Esc=quit" if char else "done -- saving")
    cv2.putText(frame, prompt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    cv2.putText(frame, harvester.status, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run_type_mode(args) -> None:
    prior = None
    if Path(args.out).exists():
        try:
            prior = keymap.KeyMap.load(args.out)
        except (OSError, json.JSONDecodeError, KeyError):
            prior = None

    harvester = Harvester(
        list(keymap.TYPING_ORDER), protected=prior,
        camera_meta={"index": args.camera, "width": args.width, "height": args.height})

    if not harvester.order:
        print("every key is already hand-edited (source='edited') -- nothing to harvest.")
        print("use --edit to change positions, or edit the JSON directly to re-open one for typing.")
        return

    print(f"harvesting {len(harvester.order)} keys x {harvester.samples_per_key} samples "
          f"(~{len(harvester.order) * harvester.samples_per_key} keystrokes, a few minutes)")
    print("press each prompted key at a normal pace, any finger. Enter = skip (mark absent). Esc = quit.")
    print("if your camera is on the far side of the keyboard, MediaPipe's left/right hand labels are "
          "likely swapped -- check the on-screen fingertip colours (cyan=left, orange=right) against "
          "your actual hands, and rerun with --flip-handedness if they're backwards.")

    tracker = HandTracker(args.model, args.flip_handedness)
    cap = open_camera(args.camera, args.width, args.height)
    if args.pin_exposure is not None:
        pin_exposure(cap, args.pin_exposure)
    fps = measure_fps(cap)
    print(f"measured capture rate: {fps:.1f} fps"
          + ("  (low -- more light or --pin-exposure will help sample quality)" if fps < 24 else ""))

    capture = TypeCapture()
    capture.start()
    cv2.namedWindow("calibrate: type to place keys")

    try:
        while not harvester.done:
            ok, frame = cap.read()
            if not ok:
                continue
            pose, _ = tracker.pose(frame, time.perf_counter())
            harvester.on_frame(pose)

            while capture.control:
                cmd = capture.control.popleft()
                if cmd == "quit":
                    raise SystemExit("cancelled -- nothing saved")
                harvester.on_control(cmd)

            while capture.queue:
                char, t = capture.queue.popleft()
                harvester.on_key(char, t)

            canvas = frame.copy()
            _draw_type_overlay(canvas, harvester, pose)
            cv2.imshow("calibrate: type to place keys", canvas)
            cv2.waitKey(1)   # pump the GUI event queue; input comes from TypeCapture
    finally:
        capture.stop()
        cap.release()
        cv2.destroyAllWindows()

    harvester.km.save(args.out)
    n_absent = sum(1 for e in harvester.km.entries.values() if e.absent)
    print(f"saved {args.out}  ({len(harvester.km.entries)} keys, {n_absent} marked absent)")


# ==================================================================== B. ===
# Stage B: the editor.


def grab_frame(camera: int, width: int, height: int) -> np.ndarray:
    cap = open_camera(camera, width, height)
    print("Position your hands away from the keyboard. Press SPACE to freeze the frame.")
    frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            preview = frame.copy()
            cv2.putText(preview, "SPACE to freeze, q to quit", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("calibrate", preview)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                break
            if k == ord("q"):
                raise SystemExit("cancelled")
    finally:
        cap.release()
    return frame


class EditState:
    def __init__(self, km: keymap.KeyMap):
        self.km = km
        self.selection: set[str] = set()
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.step = 1
        self.dragging = False
        self.drag_snapshotted = False
        self.drag_last = (0, 0)
        self.marquee_start = None
        self.marquee_end = None
        self.status = "type a key's character to select it"
        self.first_click_checked = False

    def snapshot(self) -> None:
        self.undo_stack.append(dict(self.km.entries))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            self.status = "nothing to undo"
            return
        self.redo_stack.append(dict(self.km.entries))
        self.km.entries = self.undo_stack.pop()
        self.status = "undo"

    def redo(self) -> None:
        if not self.redo_stack:
            self.status = "nothing to redo"
            return
        self.undo_stack.append(dict(self.km.entries))
        self.km.entries = self.redo_stack.pop()
        self.status = "redo"

    def touch(self, char: str, px: tuple[float, float]) -> None:
        self.km.set_pixel(char, px, source="edited")

    def move_selection(self, dx: float, dy: float) -> None:
        if not self.selection:
            self.status = "nothing selected -- type a key's character first"
            return
        self.snapshot()
        for c in self.selection:
            e = self.km.entries.get(c)
            base = e.px if (e and e.px is not None) else (0.0, 0.0)
            self.touch(c, (base[0] + dx, base[1] + dy))

    def mark_absent_selection(self) -> None:
        if not self.selection:
            return
        self.snapshot()
        for c in self.selection:
            self.km.set_absent(c)
        self.status = f"marked {len(self.selection)} key(s) absent"

    def distribute(self) -> None:
        if len(self.selection) < 3:
            self.status = "select the two end keys plus the ones between them (3+) to distribute"
            return
        ordered = sorted(self.selection, key=lambda c: keymap.ANSI_REFERENCE_XY.get(c, (0.0, 0.0))[0])
        a, b = ordered[0], ordered[-1]
        pa, pb = self.km.pixel_of(a), self.km.pixel_of(b)
        if pa is None or pb is None:
            self.status = f"both end keys ('{a}', '{b}') need a position first"
            return
        xa = keymap.ANSI_REFERENCE_XY.get(a, (0.0, 0.0))[0]
        xb = keymap.ANSI_REFERENCE_XY.get(b, (0.0, 0.0))[0]
        if xb == xa:
            self.status = "end keys have the same reference position -- can't distribute"
            return
        self.snapshot()
        for c in ordered[1:-1]:
            xc = keymap.ANSI_REFERENCE_XY.get(c, (0.0, 0.0))[0]
            frac = (xc - xa) / (xb - xa)
            px = (pa[0] + frac * (pb[0] - pa[0]), pa[1] + frac * (pb[1] - pa[1]))
            self.km.set_pixel(c, px, source="interpolated")
        self.status = f"distributed {len(ordered) - 2} key(s) between '{a}' and '{b}'"

    def _group_positions(self) -> dict[str, tuple[float, float]]:
        return {c: self.km.pixel_of(c) for c in self.selection if self.km.pixel_of(c) is not None}

    def scale_selection(self, factor: float) -> None:
        pts = self._group_positions()
        if len(pts) < 2:
            return
        cx = statistics.mean(p[0] for p in pts.values())
        cy = statistics.mean(p[1] for p in pts.values())
        self.snapshot()
        for c, (x, y) in pts.items():
            self.touch(c, (cx + (x - cx) * factor, cy + (y - cy) * factor))
        self.status = f"scaled {len(pts)} key(s) x{factor:.2f}"

    def rotate_selection(self, degrees: float) -> None:
        pts = self._group_positions()
        if len(pts) < 2:
            return
        cx = statistics.mean(p[0] for p in pts.values())
        cy = statistics.mean(p[1] for p in pts.values())
        rad = math.radians(degrees)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        self.snapshot()
        for c, (x, y) in pts.items():
            dx, dy = x - cx, y - cy
            self.touch(c, (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
        self.status = f"rotated {len(pts)} key(s) {degrees:+.0f} deg"


def _nearest_key(km: keymap.KeyMap, x: float, y: float, radius: float = HIT_RADIUS) -> str | None:
    best, best_d = None, radius
    for c, e in km.entries.items():
        if e.px is None:
            continue
        d = math.hypot(e.px[0] - x, e.px[1] - y)
        if d <= best_d:
            best_d, best = d, c
    return best


def make_mouse_callback(state: EditState, frame_shape):
    h, w = frame_shape[:2]

    def on_mouse(event, x, y, flags, _param):
        if not state.first_click_checked and event == cv2.EVENT_LBUTTONDOWN:
            state.first_click_checked = True
            if x > w * 1.3 or y > h * 1.3:
                state.status = (f"!! click at ({x},{y}) is way outside the {w}x{h} frame -- "
                                 "likely Retina mouse-scaling. Use keyboard-only controls: "
                                 "type a key to select, arrows to nudge.")

        if event == cv2.EVENT_LBUTTONDOWN:
            shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            hit = _nearest_key(state.km, x, y)
            if hit is not None:
                if ctrl:
                    state.selection.symmetric_difference_update({hit})
                elif shift:
                    state.selection.add(hit)
                elif hit not in state.selection:
                    state.selection = {hit}
                state.dragging = True
                state.drag_snapshotted = False
                state.drag_last = (x, y)
            else:
                only = next(iter(state.selection)) if len(state.selection) == 1 else None
                if only is not None and state.km.pixel_of(only) is None:
                    state.snapshot()
                    state.touch(only, (float(x), float(y)))
                    state.status = f"placed '{only}'"
                else:
                    state.marquee_start = (x, y)
                    state.marquee_end = (x, y)
                    if not shift and not ctrl:
                        state.selection = set()

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.dragging and state.selection:
                dx, dy = x - state.drag_last[0], y - state.drag_last[1]
                if dx or dy:
                    if not state.drag_snapshotted:
                        state.snapshot()
                        state.drag_snapshotted = True
                    for c in state.selection:
                        e = state.km.entries.get(c)
                        base = e.px if (e and e.px is not None) else (float(x), float(y))
                        state.touch(c, (base[0] + dx, base[1] + dy))
                    state.drag_last = (x, y)
            elif state.marquee_start is not None:
                state.marquee_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if state.marquee_start is not None:
                x0, y0 = state.marquee_start
                lo_x, hi_x = sorted((x0, x))
                lo_y, hi_y = sorted((y0, y))
                found = {c for c, e in state.km.entries.items()
                         if e.px is not None and lo_x <= e.px[0] <= hi_x and lo_y <= e.px[1] <= hi_y}
                state.selection |= found
                if found:
                    state.status = f"added {len(found)} key(s) to selection"
                state.marquee_start = None
                state.marquee_end = None
            state.dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                state.rotate_selection(5 if delta > 0 else -5)
            else:
                state.scale_selection(1.05 if delta > 0 else 1 / 1.05)

    return on_mouse


def _draw_edit_overlay(frame, state: EditState) -> None:
    km = state.km
    for c, e in km.entries.items():
        if e.px is None:
            continue
        colour = keymap.key_colour(e, km.pitch_near(c))
        x, y = int(e.px[0]), int(e.px[1])
        selected = c in state.selection
        cv2.circle(frame, (x, y), 7 if selected else 4, colour, -1)
        if selected:
            cv2.circle(frame, (x, y), 11, (0, 255, 255), 2)
        cv2.putText(frame, c, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    unplaced = [c for c in state.selection if km.pixel_of(c) is None]
    if unplaced:
        cv2.putText(frame, f"'{unplaced[0]}' selected, no position yet -- click to place",
                    (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    if state.marquee_start and state.marquee_end:
        cv2.rectangle(frame, state.marquee_start, state.marquee_end, (0, 255, 255), 1)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 62), (20, 20, 20), -1)
    cv2.putText(frame, f"{len(state.selection)} selected   step={state.step}px   {state.status}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(frame,
                "type=select  Shift+letter=add  drag/arrows=move  wheel=scale  ctrl+wheel=rotate  "
                "Backspace=absent  Ctrl+D=distribute  Ctrl+Z/Y=undo/redo  Tab=step  Enter=save  Esc=quit",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


def _load_for_edit(out_path: str) -> keymap.KeyMap:
    p = Path(out_path)
    if p.exists():
        km = keymap.KeyMap.load(out_path)
        print(f"loaded {out_path} ({len(km.entries)} keys)")
        return km
    legacy = Path("calibration.npz")
    if legacy.exists():
        km = keymap.KeyMap.load(str(legacy))
        print(f"no {out_path} yet -- loaded legacy {legacy} as a starting point "
              "(all keys are source='homography', a rough guess -- expect to fix most of them)")
        return km
    print(f"no existing calibration -- starting blank. Type a key's character to select it, "
          "then click (or use arrows) to place it.")
    return keymap.KeyMap()


def run_edit_mode(args) -> None:
    km = _load_for_edit(args.out)
    mismatch = km.check_camera(args.width, args.height)
    if mismatch:
        print(f"!! {mismatch}")

    frame = grab_frame(args.camera, args.width, args.height)
    state = EditState(km)
    cv2.namedWindow("calibrate: edit")
    cv2.setMouseCallback("calibrate: edit", make_mouse_callback(state, frame.shape))

    try:
        while True:
            canvas = frame.copy()
            _draw_edit_overlay(canvas, state)
            cv2.imshow("calibrate: edit", canvas)
            k = cv2.waitKeyEx(20)
            if k == -1:
                continue

            if k in ARROW_UP:
                state.move_selection(0, -state.step)
            elif k in ARROW_DOWN:
                state.move_selection(0, state.step)
            elif k in ARROW_LEFT:
                state.move_selection(-state.step, 0)
            elif k in ARROW_RIGHT:
                state.move_selection(state.step, 0)
            else:
                b = k & 0xFF
                if b == 27:                  # Esc
                    raise SystemExit("cancelled -- nothing saved")
                if b in (13, 10):            # Enter
                    break
                if b == 9:                   # Tab
                    state.step = 10 if state.step == 1 else 1
                    state.status = f"nudge step: {state.step}px"
                elif b in (8, 127):          # Backspace
                    state.mark_absent_selection()
                elif b == 26:                # Ctrl+Z
                    state.undo()
                elif b == 25:                # Ctrl+Y
                    state.redo()
                elif b == 4:                 # Ctrl+D
                    state.distribute()
                elif 32 <= b < 127:
                    ch = chr(b)
                    if ch.isupper():
                        norm = keymap.normalize(ch.lower())
                        if norm:
                            state.selection.add(norm)
                            state.status = f"added '{norm}' ({len(state.selection)} selected)"
                    else:
                        norm = keymap.normalize(ch)
                        if norm:
                            state.selection = {norm}
                            state.status = f"selected '{norm}'"
    finally:
        cv2.destroyAllWindows()

    km.save(args.out)
    print(f"saved {args.out}  ({len(km.entries)} keys)")


# ======================================================================= ===
def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--type", action="store_true",
                       help="bootstrap by typing (recommended for a first calibration)")
    mode.add_argument("--edit", action="store_true",
                       help="hand-place / fix individual keys (default if no mode is given)")
    ap.add_argument("--replay", metavar="PATH",
                     help="headless: harvest from a frames+events JSON fixture instead of "
                          "the camera (for testing)")
    ap.add_argument("--model", default="hand_landmarker.task")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--flip-handedness", action="store_true",
                     help="use if left/right hand labels are swapped (non-mirrored camera)")
    ap.add_argument("--pin-exposure", type=float,
                     help="lock manual exposure (backend-specific units), for --type in a dark scene")
    args = ap.parse_args()

    if args.replay:
        run_replay_mode(args.replay, out=args.out)
        return
    if args.type:
        run_type_mode(args)
    else:
        run_edit_mode(args)


if __name__ == "__main__":
    main()
