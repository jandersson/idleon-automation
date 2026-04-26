"""Smoke-test every [project.scripts] entry point.

Each entry is `module:func`. We import the module and verify the function
exists and is callable. This catches:
- Broken imports inside the module (e.g. importing a constant that was
  removed during a refactor — the bug we just hit with score_calibrate
  and SCORE_REGION_REL)
- Typos in pyproject.toml's module:func references
- Modules whose top-level code raises on import

We do NOT call the function — most need a live game window. If a function's
top-level code makes IO assumptions (e.g. tries to open a window), the
import step won't run those, only function calls would.
"""
import importlib
import sys
from pathlib import Path


def _load_entry_points() -> list[tuple[str, str, str]]:
    """Parse [project.scripts] from pyproject.toml. Returns (cmd, module, func) tuples."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # pragma: no cover

    root = Path(__file__).parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    out = []
    for cmd, target in scripts.items():
        module, _, func = target.partition(":")
        out.append((cmd, module, func))
    return out


ENTRY_POINTS = _load_entry_points()


def test_at_least_one_entry_point_defined():
    """Sanity check: pyproject.toml has scripts."""
    assert len(ENTRY_POINTS) > 0


def _ids(eps):
    return [f"{cmd}->{module}:{func}" for cmd, module, func in eps]


def pytest_generate_tests(metafunc):
    if "entry_point" in metafunc.fixturenames:
        metafunc.parametrize("entry_point", ENTRY_POINTS, ids=_ids(ENTRY_POINTS))


def test_entry_point_imports_and_resolves(entry_point):
    cmd, module_name, func_name = entry_point
    mod = importlib.import_module(module_name)
    assert hasattr(mod, func_name), (
        f"{cmd}: module '{module_name}' has no attribute '{func_name}' "
        f"(referenced as {module_name}:{func_name} in pyproject.toml)"
    )
    fn = getattr(mod, func_name)
    assert callable(fn), f"{cmd}: {module_name}:{func_name} is not callable"
