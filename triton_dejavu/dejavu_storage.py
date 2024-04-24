
import sys
import os
import json
import torch
import triton
import hashlib

from triton_dejavu import __version__ as dejavu_version


__storage_env_var__ = 'TRITON_DEJAVU_STORAGE'


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
        cuda_version = os.environ.get('VLLM_CUDA_VERSION', 'unkown')
        if cuda_version == 'unkown':
            raise Exception("Can't determine cuda version and also VLLM_CUDA_VERSION is not set")
    return cuda_version


def _get_str_signature(fn_src):
    s0 = fn_src.split()[1]
    s1 = s0.split('):')[0]
    return s1 + '):'.strip()


def _create_tuple(k):
    s = k[1:-1]
    entries = s.split(", ")
    ret = []
    for e in entries:
        try:
            ei = int(e)
            ret.append(ei)
        except ValueError:
            if type(e) == str:  # and e[1:-1][:6] == 'torch.':
                ret.append(e[1:-1])
            else:
                ret.append(e)
    ret_t = tuple(ret)
    return ret_t


__int_config_args__ = ['num_warps', 'num_stages', 'num_ctas']
__bool_config_args__ = ['enable_warp_specialization']
# __config_args__ = ['num_warps', 'num_stages', 'num_ctas', 'enable_warp_specialization', 'pre_hook']
__config_args__ = ['pre_hook'] + __int_config_args__ + __bool_config_args__
__skip_config_args__ = ['enable_persistent']


def _create_config_args(v):
    vlist = v.split(', ')
    ret = {'kwargs': {}}
    for e in vlist:
        sl = e.split(': ')
        if sl[0] in __skip_config_args__:
            continue
        if sl[0] in __config_args__:
            if sl[0] in __int_config_args__:
                ret[sl[0]] = int(sl[1])
            elif sl[0] in __bool_config_args__:
                ret[sl[0]] = bool(sl[1])
            else:
                ret[sl[0]] = sl[1]
        else:
            try:
                ret['kwargs'][sl[0]] = int(sl[1])
            except ValueError:
                try:
                    ret['kwargs'][sl[0]] = bool(sl[1])
                except ValueError:
                    print(f"[triton-dejavu] WARNING: can't determine type of kwarg {sl[0]}: {sl[1]}")
                    ret['kwargs'][sl[0]] = sl[1]
    return ret


def _get_src_hash(src):
    return hashlib.sha256(src.encode('utf-8')).hexdigest()


class DejavuStorage:

    def __init__(self) -> None:
       self.storage_prefix =  os.environ.get(__storage_env_var__, 'none')
       if self.storage_prefix == 'none':
           raise Exception(f'The environment variable {__storage_env_var__} must be set for triton-dejavu!')
       self.cuda_version = _get_cuda_version()
       self.gpu_name = torch.cuda.get_device_name().replace(' ', '_')
       self.triton_version = triton.__version__
       self.storage_path = os.path.abspath(f"{self.storage_prefix}/dejavu_{dejavu_version}/{self.cuda_version}/{self.triton_version}/{self.gpu_name}")
       os.system(f"mkdir -p {self.storage_path}")
       self.fn_storage = {}

    def __store__(self):
        for fn_hash in self.fn_storage:
            os.system(f"mkdir -p {self.storage_path}/{fn_hash}/")
            with open(f"{self.storage_path}/{fn_hash}/cache.json", 'w') as f:
                json.dump(self.fn_storage[fn_hash], f)


    def add_autotuner_cache(self, cache, fn):
        # fn.hash is not always there (apparently race condition with @triton.jit decorator?)
        # fn_hash = fn.hash
        fn_hash = _get_src_hash(fn.src)
        if fn_hash not in self.fn_storage:
            # cache_json = {'signature': _get_str_signature(fn.src)}
            cache_json = {'signature': str(fn)}
        else:
            cache_json = self.fn_storage[fn_hash]
        changes_made = False
        for key, config in cache.items():
            if str(key) in cache_json:
                continue
            cache_json[str(key)] = str(config)
            changes_made = True
            if os.environ.get("TRITON_DEJAVU_DEBUG", '0') == '1':
                print(f"[triton-dejavu] added {str(config)} for {fn_hash}")
        if changes_made:
            self.fn_storage[fn_hash] = cache_json
            self.__store__()

    def restore_autotuner_cache(self, fn):
        # fn.hash is not always there (apparently race condition with @triton.jit decorator?)
        # fn_hash = fn.hash
        fn_hash = _get_src_hash(fn.src)
        # print(fn_hash)
        cache_file = f"{self.storage_path}/{fn_hash}/cache.json"
        if not os.path.isfile(cache_file):
            return {}
        with open(cache_file, 'r') as f:
            cache_json = json.load(f)
        self.fn_storage[fn_hash] = cache_json
        ret = {}
        for k, v in cache_json.items():
            if k == 'signature':
                continue
            kt = _create_tuple(k)
            va = _create_config_args(v)
            c = triton.Config(**va)
            ret[kt] = c
            if os.environ.get("TRITON_DEJAVU_DEBUG", '0') == '1':
                print(f"[triton-dejavu] restored {str(c)} for {fn_hash}")
        return ret






global_dejavu_storage = DejavuStorage()

