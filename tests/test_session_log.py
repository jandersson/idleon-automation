"""Tests for common/session_log.py — tee stdout to a log file."""
from common.session_log import latest_log, session_log


def test_log_file_is_created_and_captures_stdout(tmp_path, capsys):
    with session_log(tmp_path) as log_path:
        print("hello")
        print("world")

    text = log_path.read_text(encoding="utf-8")
    assert "hello" in text
    assert "world" in text
    # Bracketing comments from session_log itself.
    assert "session started" in text
    assert "session ended" in text


def test_stdout_still_visible_to_user(tmp_path, capsys):
    """Tee means BOTH the log file and the original stdout get the output."""
    with session_log(tmp_path):
        print("user-facing")

    captured = capsys.readouterr()
    assert "user-facing" in captured.out


def test_latest_log_returns_most_recent(tmp_path):
    """Multiple sessions create multiple files; latest_log picks the newest."""
    assert latest_log(tmp_path) is None  # empty dir → None

    with session_log(tmp_path) as first:
        pass
    with session_log(tmp_path) as second:
        pass

    found = latest_log(tmp_path)
    assert found is not None
    assert found.name == second.name


def test_stdout_restored_after_context(tmp_path):
    import sys
    original = sys.stdout
    with session_log(tmp_path):
        assert sys.stdout is not original  # tee'd while inside
    assert sys.stdout is original  # restored after
