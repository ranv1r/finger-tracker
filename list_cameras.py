"""Find out which camera index is which.

    python list_cameras.py           # probe indices 0..5, print what answers
    python list_cameras.py --show    # also open a preview of each working one

Expect noisy warnings from the camera backend for indices that don't exist.
They're harmless -- only the summary table at the end matters.
"""
import argparse

import cv2


def probe(index: int) -> dict | None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    info = None
    if ok and frame is not None:
        h, w = frame.shape[:2]
        info = {
            "index": index,
            "resolution": f"{w}x{h}",
            "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 1),
            "frame": frame,
        }
    cap.release()
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=5)
    ap.add_argument("--show", action="store_true",
                    help="preview each working camera; any key advances")
    args = ap.parse_args()

    found = []
    for i in range(args.max_index + 1):
        info = probe(i)
        if info:
            found.append(info)

    print("\n" + "=" * 44)
    if not found:
        print("No cameras responded. Check permissions and that nothing else\n"
              "(video call, other script) is holding the device open.")
    for info in found:
        print(f"  --camera {info['index']}   {info['resolution']:>10}  "
              f"{info['fps']:>5} fps reported")
    print("=" * 44)

    if args.show:
        for info in found:
            label = f"camera {info['index']} -- press any key"
            frame = info["frame"].copy()
            cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.imshow("preview", frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()