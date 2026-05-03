"""Dump the decoded Idleon save to a JSON file for offline inspection.

Usage:
    uv run python scripts/dump_idleon_save.py [out_path]

Default output: data/idleon_save_dump.json (gitignored, like global_tries.json).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from common.idleon_save import load_save


def main() -> None:
    out_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else ROOT / "data" / "idleon_save_dump.json"
    )
    data = load_save()
    if data is None:
        print("Couldn't load save (plyvel not installed, or save dir missing).",
              file=sys.stderr)
        sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {len(data)} top-level keys ({size_mb:.1f} MB) to {out_path}")


if __name__ == "__main__":
    main()
