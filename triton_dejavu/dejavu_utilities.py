#  /*******************************************************************************
#   * Copyright 2024 -- 2025 IBM Corporation
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
if flag_print_debug:
    flag_print_autotuning = True

cuda_version = None
rocm_version = None


def _get_cuda_version():
    """
    Get CUDA runtime/driver version (i.e. which ptxas is used).
    This version is often different from the cuda version pytorch uses internally.

    Based on https://github.com/triton-lang/triton/blob/9d6736a501d0499348d48d192b6260338ca19da0/third_party/nvidia/backend/compiler.py#L32-L37
    """
    global cuda_version
    if cuda_version is not None:
        return cuda_version
    if "_TRITON_DEJAVU_DETERMINED_CUDA_VERSION" in os.environ:
        cuda_version = os.environ["_TRITON_DEJAVU_DETERMINED_CUDA_VERSION"]
        return cuda_version
    try:
        import subprocess
        import re

        triton_backend_dir = os.path.dirname(triton.backends.__file__)
        ptxas_path = os.path.abspath(
            os.path.join(triton_backend_dir, "nvidia/bin/ptxas")
        )

        result = subprocess.check_output(
            [ptxas_path, "--version"], stderr=subprocess.STDOUT
        )
        version = re.search(
            r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE
        )
        cuda_version = version.group(1)
    except Exception as e:
        if flag_print_debug:
            print(
                f"[triton-dejavu] determining cuda version failed with: {e}\n"
                f"using torch.version.cuda as fallback"
            )
        cuda_version = f"torch_{torch.version.cuda}"
    os.environ["_TRITON_DEJAVU_DETERMINED_CUDA_VERSION"] = cuda_version
    return cuda_version


def _get_rocm_version():
    """
    Get ROCM runtime/driver version (i.e. which rocm linker is used).
    This version is often different from the rocm version pytorch uses internally.
    """
    global rocm_version
    if rocm_version is not None:
        return rocm_version
    if "_TRITON_DEJAVU_DETERMINED_ROCM_VERSION" in os.environ:
        rocm_version = os.environ["_TRITON_DEJAVU_DETERMINED_ROCM_VERSION"]
        return rocm_version
    try:
        import subprocess
        import re

        rocm_ldd_path = triton.backends.backends["amd"].compiler.path_to_rocm_lld()
        rocm_dir = os.path.dirname(rocm_ldd_path)
        amdgpu_arch_path = os.path.abspath(os.path.join(rocm_dir, "amdgpu-arch"))

        result = subprocess.check_output(
            [amdgpu_arch_path, "--version"],
            stderr=subprocess.STDOUT,
        )
        version = re.search(
            r".*roc-(\d+\.\d+.\d+).*", result.decode("utf-8"), flags=re.MULTILINE
        )
        rocm_version = version.group(1)
    except Exception as e:
        if flag_print_debug:
            print(
                f"[triton-dejavu] determining rocm version failed with: {e}\n"
                f"using torch.version.hip as fallback"
            )
        rocm_version = f"torch_{torch.version.hip}"
    os.environ["_TRITON_DEJAVU_DETERMINED_ROCM_VERSION"] = rocm_version
    return rocm_version


def create_dir_if_not_exist_recursive(path, mode=0o777):
    # 0777 permissions to avoid problems with different users in containers and host system
    norm_path = os.path.normpath(path)
    paths_l = norm_path.split(os.sep)
    path_walked = f"{os.sep}"
    for p in paths_l:
        if len(p) == 0:
            continue
        path_walked = os.path.join(path_walked, p)
        create_dir_if_not_exist(path_walked, mode)


def create_dir_if_not_exist(path, mode=0o777):
    if not os.path.exists(path):
        os.mkdir(path)
        try:
            os.chmod(path, mode)
        except PermissionError as e:
            print(f"can't set permission of directory {path}: {e}")


def get_storage_prefix():
    storage_prefix = os.environ.get(__storage_env_var__, "none")
    if storage_prefix == "none":
        storage_prefix = os.getcwd()
        print(
            f"[triton-dejavu] WARNING: The environment variable {__storage_env_var__} is not set! "
            f"Using {storage_prefix} (os.getcwd()) as fallback for triton-dejavu storage."
        )
    if not os.access(storage_prefix, os.W_OK):
        raise Exception(
            f"[triton-dejavu] The path {storage_prefix} is not writeable and can not be used "
            f"as cache storage. Consider another location using the {__storage_env_var__} environment variable."
        )
    create_dir_if_not_exist_recursive(storage_prefix)
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


def get_runtime_label():
    if torch.version.hip:
        return f"rocm_{_get_rocm_version()}"
    return f"cuda_{_get_cuda_version()}"


def get_gpu_label():
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return gpu_name


def get_storage_identifier():
    # not an absolute path! (also used as keys in dictionaries)
    runtime_label = get_runtime_label()
    gpu_name = get_gpu_label()
    triton_version = triton.__version__
    dejavu_identifier = _get_dejavu_identifier()
    storage_identifier = (
        f"{dejavu_identifier}/triton_{triton_version}/{runtime_label}/gpu_{gpu_name}"
    )
    return storage_identifier


def get_tmp_storage_path():
    storage_prefix = get_storage_prefix()
    storage_tag = get_storage_tag()
    dejavu_identifier = _get_dejavu_identifier()
    tmp_path = f"{storage_prefix}/.{dejavu_identifier}-{__tmp_path_folder_name__}-{storage_tag}/"
    create_dir_if_not_exist_recursive(tmp_path)
    return tmp_path


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
