from toddler import resources as r

V100 = dict(name="Tesla V100-SXM2-32GB", total_mib=32768)


def _host(*gpus, lease=False, guard=True, cores=96, load=10.0):
    return r.HostState(tuple(gpus), cores, load, lease, guard)


def _g(i, free, foreign=0, total=32768, name="V100"):
    return r.GpuState(i, name, f"0000:{i:02d}:00.0", total, free, foreign)


def test_picks_gpu_with_most_usable_memory_and_keeps_margin():
    p = r.plan_device(_host(_g(1, 20000), _g(2, 31000)), need_mib=4000)
    assert p.device == "cuda:2" and p.cuda_visible == "2"
    assert p.budget_mib == 4000


def test_never_plans_beyond_free_minus_margin():
    g = _g(1, 10000)                       # usable = 10000 - 8192 = 1808
    assert r.plan_device(_host(g), need_mib=4000).device == "cpu"


def test_busy_gpu_is_skipped():
    p = r.plan_device(_host(_g(1, 31000, foreign=1), _g(2, 20000)), need_mib=2000)
    assert p.device == "cuda:2" and any("busy" in x for x in p.reasons)


def test_lease_or_guard_forces_cpu():
    assert r.plan_device(_host(_g(1, 31000), lease=True), 1000).device == "cpu"
    assert r.plan_device(_host(_g(1, 31000), guard=False), 1000).device == "cpu"


def test_cpu_threads_half_cores_minus_load():
    assert r.cpu_threads(96, 10.0) == 48
    assert r.cpu_threads(96, 80.0) == 16
    assert r.cpu_threads(4, 6.0) == 1


def test_back_off_when_customer_arrives_or_memory_drops():
    p = r.Placement("cuda:1", 1, 4000, "1")
    assert r.should_back_off(p, _g(1, 25000)) is None
    assert "foreign" in r.should_back_off(p, _g(1, 25000, foreign=1))
    assert "margin" in r.should_back_off(p, _g(1, 6000))
    assert r.should_back_off(r.Placement("cpu", None, 0, ""), _g(1, 10)) is None


def test_cuda_env_uses_pci_order():
    assert r.cuda_env(r.Placement("cuda:3", 3, 1, "3")) == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "3"}


def test_configured_but_missing_guard_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "GUARD_FILE", str(tmp_path / "missing.json"))
    assert r._guard_allows() is False
    monkeypatch.setattr(r, "GUARD_FILE", "")
    assert r._guard_allows() is True


def test_hanging_coordination_counts_as_lease_taken(monkeypatch, tmp_path):
    import subprocess

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("coord", 30)

    monkeypatch.setattr(r, "COORD", str(tmp_path / "coord.py"))
    monkeypatch.setattr(r.subprocess, "run", hang)
    assert r._lease_held_by_other() is True


def test_own_process_group_is_not_foreign():
    import os
    assert os.getpid() in r._own_pids()


def test_guard_names_the_missing_preflight(monkeypatch, tmp_path):
    import json

    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"rules": {"preflight_required": True, "preflight_script": str(tmp_path / "gone.sh")}}))
    monkeypatch.setattr(r, "GUARD_FILE", str(guard))
    monkeypatch.setattr(r, "PREFLIGHT", "")
    ok, why = r.guard_status()
    assert not ok and "gone.sh missing" in why
    host = r.HostState((_g(0, 20000),), 8, 0.0, False, ok, why)
    assert "gone.sh missing" in r.plan_device(host, 512).reasons[0]
    (tmp_path / "gone.sh").write_text("exit 3\n")
    assert r.guard_status() == (False, "preflight exited with 3")


def test_guard_reasons_for_malformed_file_and_timeout(monkeypatch, tmp_path):
    import json
    import subprocess

    guard = tmp_path / "guard.json"
    guard.write_text("{not json")
    monkeypatch.setattr(r, "GUARD_FILE", str(guard))
    assert r.guard_status() == (False, f"guard file {guard} malformed (no valid 'rules')")
    script = tmp_path / "pre.sh"
    script.write_text("exit 0\n")
    guard.write_text(json.dumps({"rules": {"preflight_required": True, "preflight_script": str(script)}}))
    monkeypatch.setattr(r, "PREFLIGHT", "")

    def hang(*a, **k):
        raise subprocess.TimeoutExpired("bash", 60)

    monkeypatch.setattr(r.subprocess, "run", hang)
    assert r.guard_status() == (False, "preflight timed out after 60 s")
