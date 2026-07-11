"""Create/refresh a pinnable Windows shortcut for the Idleon launcher GUI.

Builds two things:
1. ui/launcher/assets/idleon.ico — a generated pixel-art axe icon
   (only if missing; delete the file to regenerate).
2. A Desktop shortcut "Idleon Launcher.lnk" whose target is the
   windowed console-script exe (.venv/Scripts/idleon-gui.exe, from
   [project.gui-scripts] — no console window). Windows only allows
   PROGRAMS to be pinned, which is why the target is a real exe and
   not a `uv run` command line; pin it by right-clicking the shortcut
   (or the running app's taskbar button) -> Pin to taskbar.

Re-run any time (idempotent): after `uv sync` rebuilds the venv, or if
the repo moves. Usage:

    uv run python scripts/make_launcher_shortcut.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ICO = ROOT / "ui" / "launcher" / "assets" / "idleon.ico"
EXE = ROOT / ".venv" / "Scripts" / "idleon-gui.exe"
LNK_NAME = "Idleon Launcher.lnk"


def make_icon(path: Path) -> None:
    """Pixel-art axe on a dark-green rounded square (chopping colors)."""
    from PIL import Image, ImageDraw

    s = 256
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([8, 8, s - 8, s - 8], radius=48, fill=(46, 139, 87, 255))
    # Handle: thick brown diagonal from bottom-left to upper-right.
    d.line([(72, 208), (192, 88)], fill=(121, 72, 32, 255), width=26)
    d.line([(72, 208), (192, 88)], fill=(158, 100, 46, 255), width=14)
    # Axe head: steel wedge hanging left of the handle's top end.
    d.polygon([(150, 46), (196, 92), (150, 138), (108, 130), (104, 88)],
              fill=(198, 205, 212, 255))
    d.polygon([(150, 46), (196, 92), (172, 92), (138, 62)],
              fill=(230, 235, 240, 255))  # top bevel highlight
    d.polygon([(108, 130), (104, 88), (92, 96), (88, 120)],
              fill=(120, 130, 140, 255))  # cutting edge shade
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
    print(f"icon written: {path}")


def make_shortcut(lnk_dir: Path) -> Path:
    """Create the .lnk via the WScript.Shell COM object (stdlib-free)."""
    lnk = lnk_dir / LNK_NAME
    ps = f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}')
$s.TargetPath = '{EXE}'
$s.WorkingDirectory = '{ROOT}'
$s.IconLocation = '{ICO},0'
$s.Description = 'Idleon minigame bot launcher'
$s.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print(f"shortcut written: {lnk}")
    return lnk


def main() -> None:
    if not EXE.exists():
        sys.exit(f"{EXE} not found — run `uv sync` first "
                 f"(it materialises [project.gui-scripts]).")
    if not ICO.exists():
        make_icon(ICO)
    desktop = Path.home() / "Desktop"
    make_shortcut(desktop)
    print("Pin it: right-click the shortcut -> 'Pin to taskbar' (or launch "
          "it and right-click the taskbar button).")


if __name__ == "__main__":
    main()
