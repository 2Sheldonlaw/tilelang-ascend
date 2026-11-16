import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_expand_example() -> ModuleType:
    source = Path(__file__).with_name("expand_to_mhc_kernel.py")
    spec = importlib.util.spec_from_file_location("_expand_example_for_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def expand_example() -> ModuleType:
    return _load_expand_example()


@pytest.fixture(scope="module")
def isolated_kernel_cache(tmp_path_factory) -> None:
    """Give the module its own kernel cache directory (no cross-operator clobber)."""
    import tilelang.cache

    cache_dir = tmp_path_factory.mktemp("tilelang_cache")
    previous = tilelang.cache.get_cache_dir()
    tilelang.cache.set_cache_dir(str(cache_dir))
    yield
    tilelang.cache.set_cache_dir(str(previous))


def generate_expand_test_data(n0: int, n1: int, mhc_mult: int, h: int, device: str | None = None) -> dict:
    import torch

    device = "npu" if device is None else device
    torch.manual_seed(0)
    x = torch.randn(n0, n1, h, dtype=torch.bfloat16, device=device)
    o_grad = torch.randn(n0, n1, mhc_mult, h, dtype=torch.bfloat16, device=device)
    return {"x": x, "o_grad": o_grad, "mhc_mult": mhc_mult}


@pytest.mark.parametrize("n0", [1, 2])
@pytest.mark.parametrize("n1", [1024, 4096])
@pytest.mark.parametrize("mhc_mult", [4])
@pytest.mark.parametrize("h", [1280, 1920, 2560, 4224, 5120, 7168])
def test_expand_comprehensive(n0: int, n1: int, mhc_mult: int, h: int, expand_example, isolated_kernel_cache) -> None:
    import torch

    data = generate_expand_test_data(n0, n1, mhc_mult, h)

    out_tl = expand_example.expand_to_mhc_fwd(data["x"].flatten(0, -2)).reshape(n0, n1, mhc_mult, h)
    out_ref = expand_example.expand_to_mhc_fwd_ref(data["x"], mhc_mult)
    torch.testing.assert_close(out_tl, out_ref)

    x_grad_tl = expand_example.expand_to_mhc_bwd(data["o_grad"].flatten(0, -3)).reshape(n0, n1, h)

    x_ref = data["x"].clone().requires_grad_()
    out_ref_grad = expand_example.expand_to_mhc_fwd_ref(x_ref, mhc_mult)
    torch.autograd.backward([out_ref_grad], [data["o_grad"]])
    assert x_ref.grad is not None
    torch.testing.assert_close(x_grad_tl, x_ref.grad)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
