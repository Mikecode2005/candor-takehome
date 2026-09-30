"""Portability helpers.

The supplied evaluation harness reads the Brightline data files with plain
``open()`` / ``Path.read_text()``, which use the platform default encoding. On a
cp1252 Windows box that raises ``UnicodeDecodeError`` on byte 0x8d before
ingestion even starts. The data is UTF-8, so force UTF-8 for the duration of
ingestion instead of editing the supplied harness.

Usage::

    from .compat import utf8_default_encoding
    with utf8_default_encoding():
        units, deleted, edits = records.load(data_dir)
"""
import builtins
import io
from contextlib import contextmanager
from pathlib import Path

_IS_PATCHED = False


@contextmanager
def utf8_default_encoding():
    """Make unspecified text encodings default to UTF-8 while inside the block."""
    global _IS_PATCHED
    if _IS_PATCHED:
        # Nested use (e.g. chrono extraction inside load_units) is a no-op.
        yield
        return

    real_open = builtins.open
    real_io_open = io.open
    real_read_text = Path.read_text
    real_write_text = Path.write_text

    def patched_open(file, mode="r", *args, **kwargs):
        # pathlib passes buffering/encoding/errors positionally, so only inject when the
        # caller really left encoding unset.
        if "b" not in mode and "encoding" not in kwargs and len(args) < 2:
            kwargs["encoding"] = "utf-8"
        return real_open(file, mode, *args, **kwargs)

    def patched_read_text(self, encoding=None, errors=None):
        return real_read_text(self, encoding=encoding or "utf-8", errors=errors)

    def patched_write_text(self, data, encoding=None, errors=None, newline=None):
        return real_write_text(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)

    builtins.open = patched_open
    io.open = patched_open
    Path.read_text = patched_read_text
    Path.write_text = patched_write_text
    _IS_PATCHED = True
    try:
        yield
    finally:
        _IS_PATCHED = False
        builtins.open = real_open
        io.open = real_io_open
        Path.read_text = real_read_text
        Path.write_text = real_write_text


def load_units(data_dir):
    """Load the canonical unit stream using the benchmark's own delivery/edit/delete rules.

    Returns ``(units, deleted, edits)`` where each unit is a plain dict so that
    downstream modules stay dependency-free.
    """
    import sys
    harness = Path(__file__).resolve().parents[1] / "eval_harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    import records

    with utf8_default_encoding():
        units, deleted, edits = records.load(str(data_dir))
    return (
        [{"id": u.id, "record": u.record, "time": u.time, "text": u.text} for u in units],
        deleted,
        edits,
    )
