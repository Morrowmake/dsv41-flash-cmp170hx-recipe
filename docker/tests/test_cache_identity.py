import ctypes
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import cache_seed as cache

PIN = "a" * 40

@pytest.fixture
def runtime(tmp_path, monkeypatch):
    opt = tmp_path / "opt"
    (opt / "vllm-src").mkdir(parents=True)
    (opt / "venv").mkdir()
    (opt / "vllm-src/provenance.json").write_text(json.dumps({"engine_commit": PIN}))
    (opt / "venv/requirements.lock.txt").write_text("locked dependencies")
    monkeypatch.setattr(cache.importlib.metadata, "version", lambda name: "1.0")
    driver = SimpleNamespace(status=0, version=13030)
    def get_version(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = driver.version
        return driver.status
    monkeypatch.setattr(cache.ctypes, "CDLL", lambda name: SimpleNamespace(cuDriverGetVersion=get_version))
    def identify(devices, available=True):
        cuda = SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(devices),
            get_device_name=lambda i: devices[i][0],
            get_device_capability=lambda i: devices[i][1],
            get_device_properties=lambda i: SimpleNamespace(uuid="GPU-visible-" + str(i)),
        )
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
        return cache.runtime_identity(opt)
    return identify, driver

@pytest.fixture
def nvml(monkeypatch):
    class NVMLError(Exception):
        pass
    calls = []
    module = SimpleNamespace(NVMLError=NVMLError)
    def init():
        calls.append("init")
    def shutdown():
        calls.append("shutdown")
    def handle(uuid):
        calls.append(uuid)
        return uuid
    def info(handle):
        calls.append(("pci", handle))
        return SimpleNamespace(pciDeviceId=0x20C210DE)
    module.nvmlInit = init
    module.nvmlShutdown = shutdown
    module.nvmlDeviceGetHandleByUUID = handle
    module.nvmlDeviceGetPciInfo = info
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return module, calls

@pytest.mark.parametrize("names", [
    ["NVIDIA CMP 170HX"],
    ["NVIDIA CMP 170HX 64GB"],
    ["NVIDIA CMP 170HX other suffix"],
    ["NVIDIA CMP 170HX", "NVIDIA CMP 170HX 64GB"] * 2,
])
def test_real_names_keep_canonical_identity_and_key(runtime, monkeypatch, names):
    def unexpected(uuid):
        pytest.fail("PCI lookup is unnecessary for a matching name")
    monkeypatch.setattr(cache, "pci_device_id", unexpected)
    identify, _ = runtime
    canonical = identify([("NVIDIA CMP 170HX", (8, 0))])
    actual = identify([(name, (8, 0)) for name in names])
    assert actual == canonical
    assert actual["gpu_name"] == "NVIDIA CMP 170HX"
    assert actual["engine_commit"] == PIN and actual["gpu_arch"] == "sm_80"
    assert actual["driver"] == 13030
    assert cache.key(actual) == cache.key(canonical)

@pytest.mark.parametrize("pci", [0x20C210DE, 0x20C2, 0x208210DE, 0x20B010DE, 0x20C21234, None])
def test_pci_fallback(runtime, nvml, pci):
    identify, _ = runtime
    module, calls = nvml
    module.nvmlDeviceGetPciInfo = lambda handle: SimpleNamespace(pciDeviceId=pci)
    if pci == 0x20C210DE:
        actual = identify([("NVIDIA Graphics Device", (8, 0))])
        assert actual == identify([("NVIDIA CMP 170HX", (8, 0))])
    else:
        with pytest.raises(ValueError, match="restricted to CMP 170HX sm_80"):
            identify([("NVIDIA A100-SXM4-80GB", (8, 0))])
    assert calls == ["init", "GPU-visible-0", "shutdown"]

@pytest.mark.parametrize("name", ["NVIDIA CMP 170HX", "NVIDIA CMP 170HX 64GB", "NVIDIA Graphics Device"])
@pytest.mark.parametrize("capability", [(8, 6), (9, 0)])
def test_wrong_arch_rejected_before_pci(runtime, nvml, name, capability):
    identify, _ = runtime
    _, calls = nvml
    with pytest.raises(ValueError, match="restricted to CMP 170HX sm_80"):
        identify([(name, capability)])
    assert calls == []

def test_mixed_visible_cards_rejected(runtime, nvml):
    identify, _ = runtime
    module, calls = nvml
    module.nvmlDeviceGetPciInfo = lambda handle: SimpleNamespace(pciDeviceId=0x20B010DE)
    with pytest.raises(ValueError, match="restricted to CMP 170HX sm_80"):
        identify([("NVIDIA CMP 170HX 64GB", (8, 0)), ("NVIDIA A100-SXM4-80GB", (8, 0))])
    assert "GPU-visible-1" in calls and "GPU-visible-0" not in calls

@pytest.mark.parametrize("operation", ["nvmlInit", "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetPciInfo"])
def test_nvml_failure_rejects_unknown_card_and_cleans_up(runtime, nvml, operation):
    identify, _ = runtime
    module, calls = nvml
    def fail(*args):
        raise module.NVMLError("unavailable")
    setattr(module, operation, fail)
    with pytest.raises(ValueError, match="restricted to CMP 170HX sm_80"):
        identify([("NVIDIA Graphics Device", (8, 0))])
    assert ("shutdown" in calls) == (operation != "nvmlInit")

def test_missing_nvml_rejects_unknown_card(runtime, monkeypatch):
    identify, _ = runtime
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with pytest.raises(ValueError, match="restricted to CMP 170HX sm_80"):
        identify([("NVIDIA Graphics Device", (8, 0))])

@pytest.mark.parametrize("available,devices,message", [
    (False, [], "requires CUDA devices"),
    (True, [], "restricted to CMP 170HX sm_80"),
])
def test_no_visible_devices(runtime, available, devices, message):
    identify, _ = runtime
    with pytest.raises(ValueError, match=message):
        identify(devices, available)

def test_driver_error_still_rejected(runtime):
    identify, driver = runtime
    driver.status = 1
    with pytest.raises(ValueError, match="Cannot identify CUDA driver"):
        identify([("NVIDIA CMP 170HX 64GB", (8, 0))])


@pytest.mark.parametrize("form", ["bare", "prefixed", "object"])
def test_nvml_uuid_string_form(nvml, form):
    import uuid
    identifier = uuid.UUID("12345678-1234-5678-9abc-123456789abc")
    value = {"bare": str(identifier), "prefixed": "GPU-" + str(identifier), "object": identifier}[form]
    _, calls = nvml
    assert cache.pci_device_id(value) == 0x20C210DE
    assert calls == ["init", "GPU-" + str(identifier), ("pci", "GPU-" + str(identifier)), "shutdown"]
