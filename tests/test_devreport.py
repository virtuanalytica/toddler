import pytest

pytest.importorskip("torch")

from toddler.learn import devreport, generations as G, lineage as L
from toddler.learn.multitask import MultiTaskNet


def _save(reg, gen, tid, parents=()):
    net = MultiTaskNet({"cartpole": (4, 2)}, hidden=8)
    rec = G.ToddlerRecord(gen, tid, "multitask:cartpole", {"seed": 1}, 10, list(parents), [0.5, 0.6], {"cpu": "x"},
                          software={"torch": "t"})
    reg.save(net, rec)
    return rec


@pytest.fixture
def world(tmp_path):
    reg, led = G.Registry(tmp_path), L.Ledger(tmp_path)
    code, data = {"commit": "c" * 40, "dirty": False}, {"tasks": {"cartpole": "CartPole-v1"}}
    for tid in ("t1", "t2"):
        r = _save(reg, "G1", tid)
        led.born(f"G1/{tid}", r.weights_sha256, role="population", code=code, data=data, budget={"steps": 10},
                 hardware={"cpu": "x"}, software={"torch": "t"})
    r = _save(reg, "G2", "t3", parents=["G1/t1"])
    led.born("G2/t3", r.weights_sha256, role="population", code=code, data=data, budget={"steps": 10},
             hardware={"cpu": "x"}, software={"torch": "t"},
             inherits=[L.Inheritance("G1/t1", led.weights_of("G1/t1"), ("trunk",))])
    return tmp_path, led


def test_report_only_for_surviving_generations(world):
    root, led = world
    with pytest.raises(PermissionError):
        devreport.build_html(root, "G2")
    led.selected("G2", "extinct", {"why": "test"})
    with pytest.raises(PermissionError):
        devreport.build_html(root, "G2")


def test_report_contains_inheritance_integrity_and_tree(world):
    root, led = world
    led.selected("G2", "survived", {"why": "test"})
    led.derived_agent("helper", "G2/t3", "demo", "operator", weights_sha256="h")
    html = devreport.build_html(root, "G2")
    assert "G1/t1" in html and led.weights_of("G1/t1") in html          # parent and its weights hash
    assert "AFWIJKING" not in html and "intact" in html
    assert "<svg" in html and "helper" in html


def test_tampered_weights_show_up_in_the_report(world):
    root, led = world
    led.selected("G2", "survived", {})
    (root / "G2" / "t3" / "weights.pt").write_bytes(b"tampered")
    assert "AFWIJKING" in devreport.build_html(root, "G2")


def test_pdf_is_written(world, tmp_path):
    pytest.importorskip("weasyprint")
    root, led = world
    led.selected("G2", "survived", {})
    out = devreport.write_pdf(root, "G2", tmp_path / "out" / "G2.pdf")
    assert out.read_bytes()[:5] == b"%PDF-"
