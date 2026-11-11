"""Eager backend adapter for the unchanged original single-device training code.

Production uses CUDA FP32. CPU is accepted only by the adapter's mechanical
test, not by the full training entry point. No XLA package is imported here.
"""
import torch


class EagerBackend:
    def __init__(self, device):
        self.device = torch.device(device)
        self.optimizer_steps = 0

    def set_rng_state(self, seed, device=None):
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        else:
            # Test-only CPU execution; the preceding original torch.manual_seed
            # already applies this same seed before any random values are drawn.
            torch.manual_seed(seed)

    def optimizer_step(self, optimizer, barrier=False):
        optimizer.step()
        self.optimizer_steps += 1
        if barrier:
            self.wait_device_ops()

    def mark_step(self):
        # Eager operators have already been dispatched. There is no XLA graph
        # materialization step; host reads synchronize the required tensor.
        return None

    def wait_device_ops(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def save(self, state, path):
        self.wait_device_ops()
        host_state = {name: value.detach().cpu() for name, value in state.items()}
        torch.save(host_state, path)


def initialize_cuda(core):
    if not torch.cuda.is_available():
        raise RuntimeError("The complete CUDA stage requires a CUDA accelerator; no CPU or saved-state fallback exists")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    core.DEVICE = torch.device("cuda:0")
    core.xm = EagerBackend(core.DEVICE)
    return {
        "execution_backend": "cuda", "device": str(core.DEVICE),
        "device_name": torch.cuda.get_device_name(core.DEVICE),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_xla": None,
        "floating_execution_dtype": "float32", "mixed_precision": False,
        "tf32_matmul": False, "tf32_cudnn": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "backend_difference": "CUDA eager optimizer/device/save synchronization and CUDA device RNG replace XLA operations; model, loss, optimizer formulas, fixed fold protocol and numeric seed values are retained",
        "bit_identity_to_original_tpu": "not asserted: device RNG and floating-point kernels differ",
    }
