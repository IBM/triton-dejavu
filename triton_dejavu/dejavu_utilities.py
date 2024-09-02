#  /*******************************************************************************
#   * Copyright 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#

import torch
import triton
import os
import math

from triton_dejavu import __version__ as dejavu_version

__dejavu_version_major_minor_s__ = '.'.join(dejavu_version.split('.')[:2])
dejavu_version_major = int(dejavu_version.split('.')[0])
__dejavu_version_minor_s__ = dejavu_version.split('.')[1]
dejavu_version_minor = int(__dejavu_version_minor_s__)
dejavu_version_major_minor = dejavu_version_major + dejavu_version_minor/math.pow(10, len(__dejavu_version_minor_s__))

cuda_version = None


def _get_cuda_version():
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    global cuda_version
    if cuda_version is not None:
        return cuda_version
    if "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION" in os.environ:
        cuda_version = os.environ["_TRITON_DEJAVU_DETERMINED_CUDA_VERSION"]
        return cuda_version
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        import subprocess

        nvcc_output = subprocess.check_output(
            [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
        )
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = output[release_idx].split(",")[0]
        cuda_version = nvcc_cuda_version
    except Exception as e:
        if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
            print(f"[triton-dejavu] determining cuda version failed with: {e}")
        cuda_version = os.environ.get("CONTAINER_CUDA_VERSION", "unknown")
        if cuda_version == "unknown":
            raise Exception(
                "Can't determine cuda version and also CONTAINER_CUDA_VERSION is not set"
            )
    os.environ["_TRITON_DEJAVU_DETERMINED_CUDA_VERSION"] = cuda_version
    return cuda_version


def get_storage_identifier():
    runtime_cuda_version = _get_cuda_version()
    gpu_name = torch.cuda.get_device_name().replace(" ", "_")
    triton_version = triton.__version__
    torch_version = torch.__version__
    dejavu_identifier = f"dejavu_{dejavu_version}"
    if dejavu_version_major_minor >= 0.5:
        # don't let patches void collected data
        # cache file must be compatible between minor versions
        dejavu_identifier = f"dejavu_{dejavu_version_major_minor}"
    storage_identifier = f"{dejavu_identifier}/cuda_{runtime_cuda_version}/torch_{torch_version}/triton_{triton_version}/gpu_{gpu_name}"
    return storage_identifier


def get_triton_config_parameter_names():
    """To make config parameters platform independent"""
    __non_config_names__ = ['kwargs', 'pre_hook', 'all_kwargs']
    dummy_config = triton.Config(kwargs={})
    parameter_names = [s for s in dir(dummy_config) if s[0:2] != '__' and s not in __non_config_names__]
    triton_version_major_minor = '.'.join(triton.__version__.split('.')[:2])
    if triton_version_major_minor == '2.3':
        # part of the object, but not part of the __init__ parameters for triton 2.3.x
        del parameter_names[parameter_names.index('enable_persistent')]
    return parameter_names


def get_triton_config_defaults():
    dummy_config = triton.Config(kwargs={})
    parameter_names = get_triton_config_parameter_names()
    default_dict = {p: getattr(dummy_config, p) for p in parameter_names}
    return default_dict

