"""Pick the player+arm from a captured frame; save as release.png template.

Default uses the most recent frame in assets/captures/. To use a different
frame, edit FRAME_NAME below or pass --frame <name> when invoking.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region

CAPTURES = Path(__file__).parent / "assets" / "captures"
OUT = Path(__file__).parent / "assets" / "release.png"


def run():
    frames = sorted(CAPTURES.glob("capture_*.png"))
    if not frames:
        print(f"No captures in {CAPTURES}. Run darts-capture first.")
        return

    # Pick the user-specified frame if --frame is passed, else the latest.
    frame_path = frames[-1]
    if "--frame" in sys.argv:
        i = sys.argv.index("--frame") + 1
        if i < len(sys.argv):
            requested = CAPTURES / sys.argv[i]
            if requested.exists():
                frame_path = requested
            else:
                print(f"Frame not found: {requested}; falling back to {frame_path.name}")
    print(f"Using frame: {frame_path.name}")
    print("Click the top-left and bottom-right corners around the player+arm pose you want as the release template.")

    region = pick_region(image_path=frame_path, region_name="release pose")
    if region is None:
        return

    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    crop = bgr[region["top"]:region["top"] + region["height"],
               region["left"]:region["left"] + region["width"]]
    cv2.imwrite(str(OUT), crop)
    print(f"Wrote release template ({crop.shape[1]}x{crop.shape[0]}) to {OUT}")


if __name__ == "__main__":
    run()
