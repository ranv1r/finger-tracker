# Finger coach

Webcam watches your hands, matches each keystroke to the fingertip that made it,
and erases the current word if you used the wrong finger.

```
keymap.py       key -> finger (touch typing), key -> keyboard coordinates
calibrate.py    click 4 key centres once -> homography, keyboard coords -> pixels
attribution.py  the core: keystroke event -> which fingertip caused it
coach.py        capture loop, keyboard hook, verdict, correction
train.py        optional: learn the scoring weights instead of guessing them
test_attr.py    synthetic sanity check for attribution, no webcam needed
```

## Setup

```bash
pip install mediapipe opencv-python pynput numpy
# plus, only for train.py:
pip install pandas scikit-learn
```

Download `hand_landmarker.task` from the MediaPipe Hand Landmarker page
(`ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker`) into this
folder — the float16 bundle is the right one.

```bash
python test_attr.py            # verify the logic before touching a camera
python calibrate.py            # once per camera/keyboard position
python coach.py                # watch only, no erasing
python coach.py --erase        # armed
```

## Camera placement matters more than any code here

Best to worst:

1. **Overhead**, phone/webcam on an arm ~40 cm above the keyboard, lens pointing
   straight down. Minimal occlusion, near-affine geometry, homography is exact.
2. **Behind and above the screen**, tilted down over the keyboard. Workable.
3. **Laptop lid webcam.** The keyboard is at a grazing angle, the near rows hide
   the far rows, and your wrists hide everything. Expect a lot of `unsure`.

Whatever you pick, the camera must not move after calibration. Bumping the
laptop lid invalidates the homography silently — verdicts get subtly wrong
rather than obviously broken. Recalibrate when accuracy drifts.

## Platform notes

- **macOS**: `pynput` needs Accessibility *and* Input Monitoring permission for
  your terminal (System Settings → Privacy & Security). Without it, the hook
  silently receives nothing.
- **Linux/Wayland**: global keyboard hooks are largely blocked by design.
  X11 works; on Wayland you'll need `evdev` with your user in the `input` group.
- **Windows**: works as-is; run from a normal (non-elevated) shell, and note
  that hooks don't reach elevated windows.

## Tuning

Run without `--erase` until the HUD's `ok%` looks believable. Then:

| symptom | knob |
|---|---|
| left/right finger labels swapped | `--flip-handedness` |
| too many `unsure` | lower `--min-margin` |
| confident but wrong verdicts | raise `--min-margin` |
| resting fingers get blamed | raise `approach`/`motion` weight in `attribution.py` |
| verdicts lag the keystroke | shorter `lookahead`, but you lose the press peak |

`min_margin` is the safety valve. Deleting a word you typed correctly is much
more annoying than missing one you typed wrong, so bias toward abstaining.

## Known failure modes

- **Occlusion.** Your hand covers the key it's about to press. This is
  unavoidable from any single viewpoint and is the accuracy ceiling.
- **Fast typing.** Above ~80 wpm, two keystrokes can fall inside one attribution
  window and the trajectories overlap. A 30 fps camera gives ~3 frames per
  keystroke at that speed; 60 fps helps more than any algorithm change.
- **Chords.** Shift + key means the modifier hand's pinky is also moving. The
  map scores the character's finger only.
- **`flex` is the weak feature.** It assumes a downward curl, so reaching *up*
  to the number row scores negative. It's the first thing to replace with a
  learned scorer.
- **Feedback loop.** Injected backspaces are counted and swallowed in
  `KeyLogger`. If you add other synthetic keystrokes, suppress them the same way
  or the app will attribute its own typing.

## Upgrade path, roughly in value order

1. 60 fps camera, fixed exposure, good lighting on the hands. Cheapest win by
   far — attribution is limited by evidence, not by cleverness.
2. Learned weights via `train.py`, then non-linear (gradient boosting) once you
   have a few thousand keystrokes.
3. Per-keystroke sequence model: feed the whole ±150 ms window of 21×2 landmarks
   into a small 1D CNN or GRU instead of four hand-designed scalars. This is
   where you'd go if you want it to be genuinely robust, and it needs real
   labelled data.
4. Instead of erasing, log a heatmap of which keys you hit with which finger.
   Diagnosis is more useful than punishment, and it's a much softer failure mode
   when attribution is wrong.
