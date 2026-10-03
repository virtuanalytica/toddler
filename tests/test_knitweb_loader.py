import pytest

from toddler import _knitweb


def test_clear_error_when_knitweb_cannot_be_found(monkeypatch, tmp_path):
    def fail(name):
        raise ImportError(name)

    monkeypatch.setattr(_knitweb.importlib, "import_module", fail)
    monkeypatch.setenv("TODDLER_KNITWEB_SRC", str(tmp_path))
    monkeypatch.setattr(_knitweb, "_SIBLING", tmp_path / "missing")
    with pytest.raises(ImportError, match="TODDLER_KNITWEB_SRC"):
        _knitweb.load()
