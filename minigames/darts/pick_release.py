"""Pick the dart at release angle from a captured frame; save as release.png.

**Crop just the front of the dart (shaft + tip), not the whole dart.**
The dart has three regions:
  - Tail/fletching (yellow) — visually overlaps with the player's body,
    so a crop including it includes background pixels that change with
    cosmetics.
  - Hand grip — wraps around the back of the dart and changes with
    glove cosmetics.
  - Shaft (gray, horizontally banded) and tip (gray arrowhead) — these
    stick out in front of the player into clear wall airspace; same
    pixels regardless of cosmetics OR player position after teleport.

Crop the shaft+tip only. That's the dart-sprite region that template
matching can rely on across hat/gloves/clothing changes AND across the
random teleport positions after every hit.

Default uses the most recent frame in assets/captures/. Pass
`--frame <name>` to use a different one.
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
    print("Click the top-left and bottom-right corners around the DART SHAFT + TIP only.")
    print("Skip the hand, skip the yellow fletching — they overlap player sprite that")
    print("changes with cosmetics. The gray shaft + tip are always against the wall,")
    print("so they generalise across cosmetics and post-hit teleport positions.")

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
