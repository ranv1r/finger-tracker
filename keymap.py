"""Static keyboard knowledge.

Two tables:
  ANSI_REFERENCE_XY -- where each key *would* sit on a flat ANSI board, in
                        "key pitch" units (1 unit = one key pitch, x grows
                        right, y grows toward you by row). This is a FALLBACK
                        GUESS ONLY -- it assumes a flat, unsplit, standard-
                        stagger keyboard, which is wrong for curved, split, or
                        otherwise nonstandard boards. Real per-key pixel
                        positions live in a KeyMap (below), which is measured
                        per keyboard via calibrate.py, not assumed from this
                        table.
  KEY_FINGERS       -- which finger(s) touch typing says owns each key.

KEY_FINGERS is layout knowledge about touch typing and is correct regardless
of the physical keyboard's geometry. ANSI_REFERENCE_XY is only ever used to
seed a KeyMap when nothing better (typed or hand-edited positions) exists yet.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINGERS = (
    "L_pinky", "L_ring", "L_middle", "L_index", "L_thumb",
    "R_thumb", "R_index", "R_middle", "R_ring", "R_pinky",
)

# ---------------------------------------------------------------- layout ----
# Row offsets are the standard ANSI stagger, measured in key pitches.
_ROWS = (
    (0, 0.00, "`1234567890-="),
    (1, 1.50, "qwertyuiop[]\\"),
    (2, 1.75, "asdfghjkl;'"),
    (3, 2.25, "zxcvbnm,./"),
)

ANSI_REFERENCE_XY = {}
for _row, _offset, _keys in _ROWS:
    for _i, _ch in enumerate(_keys):
        ANSI_REFERENCE_XY[_ch] = (_offset + _i, float(_row))
ANSI_REFERENCE_XY[" "] = (6.5, 4.0)

# Row-by-row traversal order, for prompting the user during `calibrate.py
# --type`: walking a row at a time minimises hand travel between prompts.
TYPING_ORDER = tuple(_ch for _row, _offset, _keys in _ROWS for _ch in _keys) + (" ",)

# ------------------------------------------------------------ finger map ----
# Standard touch typing columns. Index and pinky fingers own two columns.
_COLUMNS = {
    "L_pinky":  "`1qaz",
    "L_ring":   "2wsx",
    "L_middle": "3edc",
    "L_index":  "4rfv5tgb",
    "R_index":  "6yhn7ujm",
    "R_middle": "8ik,",
    "R_ring":   "9ol.",
    "R_pinky":  "0p;/-['=]\\",
}

KEY_FINGERS = {}
for _finger, _keys in _COLUMNS.items():
    for _ch in _keys:
        KEY_FINGERS[_ch] = frozenset({_finger})
# Either thumb is fine for space.
KEY_FINGERS[" "] = frozenset({"L_thumb", "R_thumb"})

# Shifted characters fold back onto their base key.
SHIFT_MAP = dict(zip('~!@#$%^&*()_+{}|:"<>?', '`1234567890-=[]\\;\',./'))


def normalize(char: str) -> str | None:
    """Fold a typed character to its physical key, or None if we don't map it."""
    if not char:
        return None
    if char in SHIFT_MAP:
        char = SHIFT_MAP[char]
    char = char.lower()
    return char if char in KEY_FINGERS else None


def expected_fingers(char: str) -> frozenset[str]:
    """Fingers that touch typing permits for this character. Empty = unscored."""
    key = normalize(char)
    return KEY_FINGERS.get(key, frozenset()) if key else frozenset()


def ansi_reference_position(char: str) -> tuple[float, float] | None:
    """Fallback ANSI-grid guess for a key's position, in key-pitch units.

    Only used to seed a KeyMap (as source="homography") when no measured
    position exists yet. Never used directly by attribution.
    """
    key = normalize(char)
    return ANSI_REFERENCE_XY.get(key) if key else None


# Which fingers are even plausible candidates, given a target key. Restricting
# to the correct hand plus its neighbours cuts the search space and stops silly
# verdicts like "you pressed A with your right ring finger".
def candidate_fingers(char: str) -> tuple[str, ...]:
    expected = expected_fingers(char)
    if not expected:
        return ()
    if any(f.endswith("thumb") for f in expected):
        return FINGERS  # space gets hit by all sorts of things
    hand = next(iter(expected))[0]  # "L" or "R"
    return tuple(f for f in FINGERS if f.startswith(hand) and not f.endswith("thumb")) \
        + tuple(f for f in FINGERS if f[0] != hand and not f.endswith("thumb"))


# ---------------------------------------------------------------- KeyMap ----
# Where each key *actually* sits, in image pixels, for one camera + one
# physical keyboard. This is the single source of truth calibrate.py builds
# and attribution.py reads -- no ANSI geometry, no homography, no assumption
# that the board is flat or unsplit. A homography is just one (lower-
# confidence) way to populate it; see `from_homography`.

VALID_SOURCES = ("typed", "edited", "homography", "interpolated")
_FALLBACK_PITCH_PX = 60.0   # used only when too few keys are calibrated to measure a real pitch
FORMAT_VERSION = 2


@dataclass
class KeyEntry:
    px: tuple[float, float] | None = None
    absent: bool = False
    source: str = "edited"
    samples: int = 0
    spread_px: float | None = None

    def to_dict(self) -> dict:
        if self.absent:
            return {"absent": True}
        return {
            "px": [self.px[0], self.px[1]],
            "source": self.source,
            "samples": self.samples,
            "spread_px": self.spread_px,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyEntry":
        if d.get("absent"):
            return cls(px=None, absent=True, source=d.get("source", "edited"))
        px = d["px"]
        return cls(px=(float(px[0]), float(px[1])), absent=False,
                    source=d.get("source", "edited"),
                    samples=int(d.get("samples", 0)),
                    spread_px=d.get("spread_px"))


class KeyMap:
    """Where each key is, in image pixels, for this camera and this keyboard."""

    def __init__(self, camera: dict | None = None, notes: str = "",
                 created: str | None = None):
        self.entries: dict[str, KeyEntry] = {}
        self.camera: dict = dict(camera or {})
        self.notes = notes
        self.created = created or time.strftime("%Y-%m-%dT%H:%M:%S")

    # ------------------------------------------------------------- writes --
    def set_pixel(self, char: str, px: tuple[float, float], source: str,
                  samples: int = 0, spread_px: float | None = None) -> None:
        if source not in VALID_SOURCES:
            raise ValueError(f"unknown source {source!r}")
        key = normalize(char) or char
        self.entries[key] = KeyEntry(px=(float(px[0]), float(px[1])), absent=False,
                                      source=source, samples=samples, spread_px=spread_px)

    def set_absent(self, char: str) -> None:
        key = normalize(char) or char
        self.entries[key] = KeyEntry(px=None, absent=True, source="edited")

    # -------------------------------------------------------------- reads --
    def pixel_of(self, char: str) -> tuple[float, float] | None:
        e = self.entries.get(normalize(char) or char)
        if e is None or e.absent or e.px is None:
            return None
        return e.px

    def is_calibrated(self, char: str) -> bool:
        return self.pixel_of(char) is not None

    def pitch_near(self, char: str) -> float:
        """Local key spacing in px, for normalising distances.

        Median distance to the 4 nearest *other* calibrated keys. Falls back
        to a global median (or a fixed constant, if too little is calibrated
        yet) so this never assumes uniform pitch.
        """
        key = normalize(char) or char
        calibrated = {k: e.px for k, e in self.entries.items()
                      if e.px is not None and not e.absent}
        if len(calibrated) < 2:
            return _FALLBACK_PITCH_PX

        target = calibrated.get(key)
        if target is None:
            # Unknown/uncalibrated key: fall back to the map's overall
            # nearest-neighbour spacing rather than any one key's.
            nearest = []
            for k, px in calibrated.items():
                others = sorted(math.hypot(px[0] - p2[0], px[1] - p2[1])
                                 for k2, p2 in calibrated.items() if k2 != k)
                if others:
                    nearest.append(others[0])
            return statistics.median(nearest) if nearest else _FALLBACK_PITCH_PX

        dists = sorted(math.hypot(target[0] - px[0], target[1] - px[1])
                        for k, px in calibrated.items() if k != key)
        nearest4 = dists[:4]
        return statistics.median(nearest4) if nearest4 else _FALLBACK_PITCH_PX

    def check_camera(self, width: int, height: int) -> str | None:
        """Return a warning string if this map was built for a different
        capture resolution than the one currently in use, else None. A
        resolution mismatch silently scales every position wrong -- warn
        loudly rather than let that pass."""
        cw, ch = self.camera.get("width"), self.camera.get("height")
        if cw is None or ch is None:
            return None
        if (cw, ch) != (width, height):
            return (f"calibration was captured at {cw}x{ch} but the live "
                    f"capture is {width}x{height} -- key positions will be "
                    "wrong. Recalibrate at this resolution.")
        return None

    # ------------------------------------------------------------- (de)ser --
    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "created": self.created,
            "camera": self.camera,
            "notes": self.notes,
            "keys": {k: e.to_dict() for k, e in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyMap":
        km = cls(camera=d.get("camera"), notes=d.get("notes", ""),
                  created=d.get("created"))
        for k, ed in d.get("keys", {}).items():
            km.entries[k] = KeyEntry.from_dict(ed)
        return km

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KeyMap":
        p = Path(path)
        if p.suffix == ".npz":
            return cls._load_v1_npz(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_homography(cls, H, camera: dict | None = None,
                         keys: dict[str, tuple[float, float]] | None = None) -> "KeyMap":
        """Populate a KeyMap by projecting the flat ANSI reference grid (or a
        caller-supplied keyboard-coordinate table) through a homography.
        Lowest-confidence source -- a planar transform can't model a curved
        or split board, this is a starting point for --edit, not an answer.
        """
        keys = keys if keys is not None else ANSI_REFERENCE_XY
        H = np.asarray(H, dtype=np.float64)
        km = cls(camera=camera)
        for name, (x, y) in keys.items():
            vec = H @ np.array([x, y, 1.0])
            if abs(vec[2]) < 1e-9:
                continue
            km.set_pixel(name, (vec[0] / vec[2], vec[1] / vec[2]), source="homography")
        return km

    @classmethod
    def _load_v1_npz(cls, path: str) -> "KeyMap":
        npz = np.load(path)
        H = npz["H"]
        h, w = (int(x) for x in npz["frame_shape"])
        km = cls.from_homography(H, camera={"width": w, "height": h})
        km.notes = f"converted from legacy {path}"
        return km


# BGR colours, shared by coach.py's live HUD and calibrate.py's editor, so a
# key's trustworthiness reads the same way in both tools.
def key_colour(entry: KeyEntry | None, pitch: float) -> tuple[int, int, int]:
    if entry is None:
        return (90, 90, 90)          # never touched
    if entry.absent:
        return (60, 60, 60)          # deliberately absent
    if entry.source == "edited":
        return (255, 255, 255)       # hand-placed: fully trusted
    if entry.source == "homography":
        return (0, 140, 255)         # planar guess: least trusted
    if entry.source == "interpolated":
        return (200, 130, 255)       # distributed between two anchors
    if entry.source == "typed":
        tight = (entry.spread_px or 0.0) <= 0.15 * max(pitch, 1e-6)
        return (80, 230, 120) if tight else (0, 210, 255)   # tight green / loose amber
    return (200, 200, 200)
