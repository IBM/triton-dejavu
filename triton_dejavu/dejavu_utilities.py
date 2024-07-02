import torch
import triton
import os

from triton_dejavu import __version__ as dejavu_version


def _get_cuda_version():
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        import subprocess
        nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                              universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = output[release_idx].split(",")[0]
        cuda_version = nvcc_cuda_version
    except Exception as e:
        print(f"[INFO] determining cuda version failed with: {e}")
        cuda_version = os.environ.get('CONTAINER_CUDA_VERSION', 'unkown')
        if cuda_version == 'unkown':
            raise Exception("Can't determine cuda version and also CONTAINER_CUDA_VERSION is not set")
    return cuda_version


def get_storage_identifier():
    runtime_cuda_version = _get_cuda_version()
    gpu_name = torch.cuda.get_device_name().replace(' ', '_')
    triton_version = triton.__version__
    torch_version = torch.__version__
    storage_identifier = f"dejavu_{dejavu_version}/cuda_{runtime_cuda_version}/torch_{torch_version}/triton_{triton_version}/gpu_{gpu_name}"
    return storage_identifier

