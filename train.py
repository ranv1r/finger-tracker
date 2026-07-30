"""Replace the hand-tuned weights with a learned scorer.

How to get labels cheaply: record a session in which you deliberately type with
the *correct* fingers. Then "which finger pressed the key" == "which finger the
touch-typing map expects", and every row is labelled for free.

    python coach.py --record clean.csv     # type carefully for ~5 minutes
    python train.py clean.csv

The task is framed as ranking, not 10-way classification: score each candidate
finger independently with P(this finger is the one), then argmax over the
candidates for that keystroke. This keeps the feature vector fixed-size and
handles keystrokes where only some fingers were visible.

Caveat worth knowing: a clean-typing session contains no genuine mistakes, so
the model only ever learns "which finger moved onto the key" -- which is exactly
what we want it to learn, but it means your accuracy number here is optimistic
about the wrong-finger case. Validate that separately by typing a session with
deliberate errors and checking the verdicts by hand.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ["prox", "approach", "motion", "flex"]


def main(path: str) -> None:
    df = pd.read_csv(path)
    X, y, groups = df[FEATURES].to_numpy(), df["is_expected"].to_numpy(), df["event_id"].to_numpy()

    # Split by event, never by row -- rows from one keystroke are not independent.
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                                random_state=0).split(X, y, groups))
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X[train_idx], y[train_idx])

    # Evaluate the way it will actually be used: argmax within each keystroke.
    test = df.iloc[test_idx].copy()
    test["score"] = model.decision_function(X[test_idx])
    picked = test.loc[test.groupby("event_id")["score"].idxmax()]
    top1 = picked["is_expected"].mean()

    print(f"{len(df)} rows / {df['event_id'].nunique()} keystrokes")
    print(f"top-1 finger accuracy on held-out keystrokes: {top1:.1%}")
    print("\nlearned weights (paste into attribution.DEFAULT_WEIGHTS):")
    print("{")
    for name, w in zip(FEATURES, model.coef_[0]):
        print(f'    "{name}": {w:+.3f},')
    print("}")
    print("\nfeature std devs (large spread => consider standardising):")
    print(dict(zip(FEATURES, np.round(X.std(axis=0), 3))))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
