
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
            try:
                if type(e) == str and (e == 'True' or e == 'False'):
                    eb = bool(e)
                    ret.append(eb)
                elif type(e) == str:  # and e[1:-1][:6] == 'torch.':
                    ret.append(e[1:-1])
                else: 
                    raise ValueError
            except ValueError:
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


def _get_folder_name(fn_name, fn_hash, configs_hash):
    return f"{fn_name}-{fn_hash}-{configs_hash}"


def get_config_list_hash(configs):
    # s = str(configs).encode('utf-8')
    # need to be the same hash between different program runs, so can't use "object at..."
    # sorted_configs = configs.sort()
    # order of config list should be the same if come from ConfigSpace
    # if manual, then can't be guaranteed
    #  (but may not matter in practice, since then also the source code may has changed?)
    s = '|'
    for c in configs:
        s += f"{c}|"
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return h


class DejavuStorage:

    def __init__(self) -> None:
       self.storage_prefix =  os.environ.get(__storage_env_var__, 'none')
       if self.storage_prefix == 'none':
           raise Exception(f'The environment variable {__storage_env_var__} must be set for triton-dejavu!')
       self.cuda_version = _get_cuda_version()
       self.gpu_name = torch.cuda.get_device_name().replace(' ', '_')
       self.triton_version = triton.__version__
       self.storage_identifier = f"dejavu_{dejavu_version}/{self.cuda_version}/{self.triton_version}/{self.gpu_name}"
       self.storage_path = os.path.abspath(f"{self.storage_prefix}/{self.storage_identifier}/")
       os.system(f"mkdir -p {self.storage_path}")
       self.fn_storage = {}
       self.measured_timings = {}
       self._known_files = []

    def __store__(self):
        for folder_name in self.fn_storage:
            os.system(f"mkdir -p {self.storage_path}/{folder_name}/")
            file_name = f"{self.storage_path}/{folder_name}/cache.json"
            if file_name not in self._known_files:
                self._known_files.append(file_name)
            with open(file_name, 'w') as f:
                json.dump(self.fn_storage[folder_name], f, indent=4)


    def add_autotuner_cache(self, cache, fn, configs_hash, configs_len, timings, repetitiont, warmupt, bench_time):
        # fn.hash is not always there (apparently race condition with @triton.jit decorator?)
        # fn_hash = fn.hash
        fn_hash = _get_src_hash(fn.src)
        fn_name = str(fn).split(":")[1][:-1]
        folder_name = _get_folder_name(fn_name, fn_hash, configs_hash)
        if folder_name not in self.fn_storage:
            # cache_json = {'signature': _get_str_signature(fn.src)}
            cache_json = {'signature': str(fn), 'total_bench_time_s': 0.0, 'evaluated_configs': configs_len, 
                          'cache': {}, 'timings': {}}
        else:
            cache_json = self.fn_storage[folder_name]
        changes_made = False
        for key, config in cache.items():
            if str(key) in cache_json['cache']:
                continue
            cache_json['cache'][str(key)] = str(config)
            nt = {'values': timings[key], 'lables': ['ms', 'min_ms', 'max_ms'], 'rep_t_ms': repetitiont, 'warmup_t_ms': warmupt}
            cache_json['timings'][str(key)] = nt
            changes_made = True
            if os.environ.get("TRITON_DEJAVU_DEBUG", '0') == '1':
                print(f"[triton-dejavu] added {str(config)} for {fn_hash}")
        if changes_made:
            cache_json['total_bench_time_s'] += bench_time
            self.fn_storage[folder_name] = cache_json
            self.__store__()

    def restore_autotuner_cache(self, fn, configs_hash):
        # fn.hash is not always there (apparently race condition with @triton.jit decorator?)
        # fn_hash = fn.hash
        fn_hash = _get_src_hash(fn.src)
        # print(fn_hash)
        fn_name = str(fn).split(":")[1][:-1]
        folder_name = _get_folder_name(fn_name, fn_hash, configs_hash)
        cache_file = f"{self.storage_path}/{folder_name}/cache.json"
        if not os.path.isfile(cache_file):
            return {}
        if cache_file not in self._known_files:
            self._known_files.append(cache_file)
        with open(cache_file, 'r') as f:
            cache_json = json.load(f)
        self.fn_storage[folder_name] = cache_json
        ret = {}
        # for k, v in cache_json.items():
        #     if k in ['signature', 'total_bench_time_s']:
        #         continue
        for k, v in cache_json['cache'].items():
            kt = _create_tuple(k)
            va = _create_config_args(v)
            c = triton.Config(**va)
            ret[kt] = c
            if os.environ.get("TRITON_DEJAVU_DEBUG", '0') == '1':
                print(f"[triton-dejavu] restored {str(c)} for {fn_hash}")
        return ret
    
    def dump_storage(self, filter_timings=False):
        print(f"DejavuStorage: {self.storage_identifier}")
        if filter_timings:
            tmp_json = {}
            for k,v in self.fn_storage.items():
                tmp_d = {}
                for kk, vv in v.items():
                    if kk == 'timings':
                        continue
                    tmp_d[kk] = vv
                tmp_json[k] = tmp_d
            print(json.dumps(tmp_json, indent=4))
        else:
            print(json.dumps(self.fn_storage, indent=4))


global_dejavu_storage = DejavuStorage()

