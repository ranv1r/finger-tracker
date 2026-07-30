"""The interesting part: deciding which fingertip caused a keystroke.

Given a ring buffer of timestamped hand poses and a key event at time t, we
score every plausible finger on four features and take the argmax -- unless the
margin is thin, in which case we abstain.

The four features, all designed to be independent of camera angle:

  prox      minimum distance (in local key pitches) from the tip to the
            pressed key's centre, over a window around t.         lower = better
  approach  how much that distance shrank over the ~120 ms before t.
            A resting finger has approach ~= 0.                  higher = better
  motion    path length of the tip over the window, normalised by hand size.
            The pressing finger is the one that actually moved.   higher = better
  flex      how much the tip pulled in toward its own knuckle, i.e. the finger
            curling to strike.                                    higher = better

Everything lives in image pixels end to end -- fingertips are never projected
into a keyboard coordinate space. `prox` and `approach` are pixel distances
divided by keymap.KeyMap.pitch_near(char), which keeps their units comparable
to the old key-pitch scale without assuming any particular board geometry.

`prox` and `approach` require a KeyMap with the pressed key calibrated; when
the key isn't calibrated (or no KeyMap is given), attribution abstains outright
rather than guessing from motion alone -- see `use_position_features`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import keymap

TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

FEATURE_NAMES = ("prox", "approach", "motion", "flex")

# DEFAULT_WEIGHTS = {
#     "prox": -1.0,
#     "approach": 0.8,
#     "motion": 0.5,
#     "flex": 0.4,
# }
DEFAULT_WEIGHTS = {
    "prox": -6.139,
    "approach": +0.823,
    "motion": -0.369,
    "flex": -0.002,
}

@dataclass
class FingerPose:
    tip_px: tuple[float, float]
    flexion: float                       # |tip - mcp| / hand_scale


@dataclass
class PoseFrame:
    t: float
    hand_scale: float
    fingers: dict[str, FingerPose] = field(default_factory=dict)


@dataclass
class Verdict:
    finger: str | None
    margin: float
    scores: dict[str, float]
    features: dict[str, dict[str, float]]
    reason: str = ""


class Attributor:
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        lookback: float = 0.15,
        lookahead: float = 0.06,
        press_window: float = 0.12,
        min_margin: float = 0.35,
        min_frames: int = 3,
        use_position_features: bool = True,
    ):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.lookback = lookback
        self.lookahead = lookahead
        self.press_window = press_window
        self.min_margin = min_margin
        self.min_frames = min_frames
        # False during `calibrate.py --type`: positions are what we're trying
        # to measure, so prox/approach can't be used yet -- score on motion +
        # flex only. True is the normal runtime mode.
        self.use_position_features = use_position_features

    # -------------------------------------------------------------- helpers --
    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _features(self, samples, char: str, keymap_map) -> dict[str, float]:
        """samples: list[(t, FingerPose, hand_scale)] sorted by t, spanning the window."""
        t_end = samples[-1][0]

        # ---- motion: normalised path length in image space
        path = 0.0
        for (_, a, scale), (_, b, _s) in zip(samples, samples[1:]):
            path += self._dist(a.tip_px, b.tip_px) / max(scale, 1e-6)

        # ---- flex: curl between the start of the press window and the event
        pre = [s for s in samples if s[0] <= t_end - self.press_window] or [samples[0]]
        flex = pre[-1][1].flexion - samples[-1][1].flexion

        # ---- proximity + approach, in pixels normalised by local key pitch
        prox = approach = 0.0
        if self.use_position_features and keymap_map is not None:
            key_px = keymap_map.pixel_of(char)
            pitch = keymap_map.pitch_near(char)
            if key_px is not None and pitch > 0:
                dists = [(t, self._dist(p.tip_px, key_px) / pitch) for t, p, _ in samples]
                prox = min(d for _, d in dists)
                first = dists[0][1]
                at_event = min(dists, key=lambda td: abs(td[0] - t_end))[1]
                approach = first - at_event

        return {"prox": prox, "approach": approach, "motion": path, "flex": flex}

    # ----------------------------------------------------------------- main --
    def attribute(self, buffer, t_event: float, char: str, keymap_map=None,
                  extra_margin: float = 0.0) -> Verdict:
        window = [f for f in buffer
                  if t_event - self.lookback <= f.t <= t_event + self.lookahead]
        if len(window) < self.min_frames:
            return Verdict(None, 0.0, {}, {}, "not enough frames (hands not visible?)")

        candidates = keymap.candidate_fingers(char)
        if not candidates:
            return Verdict(None, 0.0, {}, {}, "key not in finger map")

        if self.use_position_features:
            if keymap_map is None or not keymap_map.is_calibrated(char):
                return Verdict(None, 0.0, {}, {}, "key not calibrated")

        features: dict[str, dict[str, float]] = {}
        scores: dict[str, float] = {}
        for finger in candidates:
            samples = [(f.t, f.fingers[finger], f.hand_scale)
                       for f in window if finger in f.fingers]
            if len(samples) < self.min_frames:
                continue
            feats = self._features(samples, char, keymap_map)
            features[finger] = feats
            scores[finger] = sum(self.weights[k] * feats[k] for k in FEATURE_NAMES)

        if not scores:
            return Verdict(None, 0.0, scores, features, "no candidate finger tracked")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        margin = best_score - (ranked[1][1] if len(ranked) > 1 else best_score - 1.0)

        # extra_margin lets a caller demand more confidence for a specific
        # event -- coach.py uses this to abstain more readily on keys whose
        # calibrated position has a wide sample spread.
        if margin < self.min_margin + extra_margin:
            return Verdict(None, margin, scores, features,
                           f"too close to call ({best} vs {ranked[1][0]})")
        return Verdict(best, margin, scores, features)
