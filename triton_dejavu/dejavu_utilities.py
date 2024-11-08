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


__storage_env_var__ = "TRITON_DEJAVU_STORAGE"
__tag_env_var__ = "TRITON_DEJAVU_TAG"
__tag_default__ = "default"

__tmp_path_folder_name__ = "tmp"
__dejavu_version_major_minor_s__ = ".".join(dejavu_version.split(".")[:2])
dejavu_version_major = int(dejavu_version.split(".")[0])
__dejavu_version_minor_s__ = dejavu_version.split(".")[1]
dejavu_version_minor = int(__dejavu_version_minor_s__)
dejavu_version_major_minor = dejavu_version_major + dejavu_version_minor / math.pow(
    10, len(__dejavu_version_minor_s__)
)

flag_print_autotuning = os.environ.get("TRITON_PRINT_AUTOTUNING", None) == "1"
flag_print_debug = os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1"
flag_print_debug_verbose = os.environ.get("TRITON_DEJAVU_DEBUG_DEBUG", "0") == "1"
if flag_print_debug_verbose:
    flag_print_debug = True

cuda_version = None
rocm_version = None


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
        cuda_version = output[release_idx].split(",")[0]
    except Exception as e:
        if flag_print_debug:
            print(f"[triton-dejavu] determining cuda version failed with: {e}")
        cuda_version = os.environ.get("CONTAINER_CUDA_VERSION", "unknown")
        if cuda_version == "unknown":
            raise Exception(
                "Can't determine cuda version and also CONTAINER_CUDA_VERSION is not set"
            )
    os.environ["_TRITON_DEJAVU_DETERMINED_CUDA_VERSION"] = cuda_version
    return cuda_version


def _get_rocm_version():
    global rocm_version
    if rocm_version is not None:
        return rocm_version
    if "_TRITON_DEJAVU_DETERMINED_ROCM_VERSION" in os.environ:
        rocm_version = os.environ["_TRITON_DEJAVU_DETERMINED_ROCM_VERSION"]
        return rocm_version
    try:
        from torch.utils.cpp_extension import ROCM_HOME
        import subprocess

        hipcc_output = subprocess.check_output(
            [ROCM_HOME + "/bin/hipcc", "--version"], universal_newlines=True
        )
        output = hipcc_output.split()
        release_idx = output.index("HIP") + 2
        rocm_version_l = output[release_idx].split("-")[0].split(".")[:2]
        rocm_version = ".".join(rocm_version_l)
    except Exception as e:
        if flag_print_debug:
            print(f"[triton-dejavu] determining rocm version failed with: {e}")
        cuda_version = os.environ.get("CONTAINER_ROCM_VERSION", "unknown")
        if cuda_version == "unknown":
            raise Exception(
                "Can't determine cuda version and also CONTAINER_ROCM_VERSION is not set"
            )
    os.environ["_TRITON_DEJAVU_DETERMINED_ROCM_VERSION"] = rocm_version
    return rocm_version


def get_storage_prefix():
    storage_prefix = os.environ.get(__storage_env_var__, "none")
    if storage_prefix == "none":
        raise Exception(
            f"[triton-dejavu] The environment variable {__storage_env_var__} must be set for triton-dejavu!"
        )
    dir_name = os.path.dirname(storage_prefix)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, 0o0777)
    return storage_prefix


def get_storage_tag():
    # so it could change during execution...
    storage_tag = os.environ.get(__tag_env_var__, __tag_default__)
    return storage_tag


def _get_dejavu_identifier():
    dejavu_identifier = f"dejavu_{dejavu_version}"
    if dejavu_version_major_minor >= 0.5:
        # don't let patches void collected data
        # cache file must be compatible between minor versions
        dejavu_identifier = f"dejavu_{dejavu_version_major_minor}"
    return dejavu_identifier


def get_storage_identifier():
    # not an absolute path!
    if torch.version.hip:
        runtime_cuda_version = f"rocm_{_get_rocm_version()}"
    else:
        runtime_cuda_version = f"cuda_{_get_cuda_version()}"
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    triton_version = triton.__version__
    torch_version = torch.__version__
    dejavu_identifier = _get_dejavu_identifier()
    storage_identifier = f"{dejavu_identifier}/{runtime_cuda_version}/torch_{torch_version}/triton_{triton_version}/gpu_{gpu_name}"
    return storage_identifier


def get_tmp_storage_path():
    storage_prefix = get_storage_prefix()
    storage_tag = get_storage_tag()
    dejavu_identifier = _get_dejavu_identifier()
    storage_identifier = f"{storage_prefix}/{dejavu_identifier}-{__tmp_path_folder_name__}-{storage_tag}/"
    dir_name = os.path.dirname(storage_identifier)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, 0o0777)
    return storage_identifier


def get_triton_config_parameter_names():
    """To make config parameters platform independent"""
    __non_config_names__ = ["kwargs", "pre_hook", "all_kwargs"]
    dummy_config = triton.Config(kwargs={})
    parameter_names = [
        s for s in dir(dummy_config) if s[0:2] != "__" and s not in __non_config_names__
    ]
    triton_version_major_minor = ".".join(triton.__version__.split(".")[:2])
    if triton_version_major_minor == "2.3":
        # part of the object, but not part of the __init__ parameters for triton 2.3.x
        del parameter_names[parameter_names.index("enable_persistent")]
    return parameter_names


def get_triton_config_defaults_values():
    dummy_config = triton.Config(kwargs={})
    parameter_names = get_triton_config_parameter_names()
    default_dict = {p: getattr(dummy_config, p) for p in parameter_names}
    return default_dict


def get_type_dict(value_dict):
    type_dict = {}
    # type_dict should have callable members
    for k, v in value_dict.items():
        if isinstance(v, int):
            type_dict[k] = int
        elif isinstance(v, bool):
            type_dict[k] = bool
        elif isinstance(v, type(None)):
            type_dict[k] = lambda ignore: None
        else:
            # TODO: other types possible?
            type_dict[k] = str
    return type_dict


def get_triton_config_defaults_types():
    default_dict = get_triton_config_defaults_values()
    type_dict = get_type_dict(default_dict)
    return type_dict
