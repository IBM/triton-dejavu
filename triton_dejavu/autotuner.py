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

from __future__ import annotations

import builtins
import sys
import os
import time
import inspect
from typing import Dict
import itertools
import torch
import gc
import traceback

# from triton.testing import do_bench, do_bench_cudagraph
from triton.testing import do_bench as upstream_do_bench
from triton.testing import do_bench_cudagraph as upstream_do_bench_cudagraph
from triton_dejavu.testing import do_bench, do_bench_cudagraph, KernelEvalCall
from triton import KernelInterface, Config, OutOfResources

from triton import __version__ as triton_version

triton_major_version = int(triton_version.split(".")[0])

from triton_dejavu.dejavu_storage import (
    global_dejavu_storage,
    get_config_list_hash,
    get_list_hash,
    get_string_hash,
)

# TODO: make dynamic
__additional_config_arg_check__ = ['num_warps', 'num_stages', 'num_ctas']

# to be compatible with different triton 3.x versions
def _all_kwargs(self):
    if not hasattr(self, "maxnreg"):
        self.maxnreg = None
    return {
        **self.kwargs,
        **{
            k: v
            for (k, v) in (
                ("num_warps", self.num_warps),
                ("num_ctas", self.num_ctas),
                ("num_stages", self.num_stages),
                ("maxnreg", self.maxnreg),
            )
            if v is not None
        },
    }


class Autotuner(KernelInterface):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook,
        post_hook,
        prune_configs_by: Dict = None,
        warmup=5,
        rep=50,
        use_cuda_graph=False,
        config_space: ConfigSpace = None,
        use_bo = False,
        fallback_heuristic = None,
    ):
        if config_space:
            self.config_space = config_space
            assert not configs, "can't configure configs and config_space"
            self.configs = self.config_space.generate_config_list()
        else:
            self.config_space = None
            if not configs:
                self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
            else:
                self.configs = configs
        self.configs_hash = get_config_list_hash(self.configs)
        self.run_id = 0
        self._obj_hash = hash(self)
        # the key hash is not covered by fn.hash!
        self.key_hash = get_list_hash(key)
        self.configs_len = len(self.configs)
        self.config_kw_names = list(self.configs[0].kwargs.keys())
        self.key_idx = [arg_names.index(k) for k in key]
        self.arg_names = arg_names

        # Reset to zero or restore values
        self.reset_idx = []
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]
        self.restore_idx = []
        if restore_value is not None:
            self.restore_idx = [arg_names.index(k) for k in restore_value]
        self.restore_copies = []

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda args, reset_only=False: 0
        self.post_hook = lambda args: 0
        self.custom_pre_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.custom_pre_hook = True
        elif len(self.reset_idx) > 0 or len(self.restore_idx) > 0:

            def _pre_hook(args, reset_only=False):
                for i in self.reset_idx:
                    args[i].zero_()
                if not reset_only:
                    self.restore_copies = [args[i].clone() for i in self.restore_idx]

            self.pre_hook = _pre_hook

        self.custom_post_hook = False
        if post_hook:
            self.post_hook = post_hook
            self.custom_post_hook = True
        elif len(self.restore_idx) > 0:

            def _post_hook(args):
                for i, j in enumerate(self.restore_idx):
                    args[j].copy_(self.restore_copies[i])
                # is apparently unrelated...
                # del self.restore_copies  # to be sure...?
                self.restore_copies = []

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get(
                "early_config_prune", self.early_config_prune
            )
            print(
                "[Triton Dejavu:WARNING] use of 'prune_configs_by' could influence the autotuner decision in a way not visible to triton-dejavu. Please ensure that configs could be reused."
            )
        # TODO: how to include in param hash?

        self.warmup_t = warmup
        self.rep_t = rep

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self._timings = {}
        if triton_major_version >= 3:
            self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
            self.benchmarking_stream = (
                torch.cuda.Stream() if self.use_cuda_graph else None
            )
        else:
            self.use_cuda_graph = False
            self.benchmarking_stream = None

        self.use_bo = use_bo
        if self.use_bo:
            if not self.config_space or prune_configs_by:
                raise Exception(f"[triton-dejavu] BO search can only be used in combination with config_space " 
                                f"and without prune_configs_by!")
            # doesn't work... (also not if done before the other imports)
            # from smac.utils.logging import setup_logging
            # setup_logging(40)
            # test imports 
            from ConfigSpace import Configuration, ConfigurationSpace
            import numpy as np
            from smac import HyperparameterOptimizationFacade, Scenario
            # convert config space
            self.bohb_config_space = self.config_space.get_BohbConfigSpace()
            # use 2% as starting point...
            share_of_trials_as_max = float(os.environ.get('TRITON_DEJAVU_BO_MAX_TRIAL_SHARE', 0.02))
            self.bohb_max_n_trials = int(min(max(50, share_of_trials_as_max * self.configs_len), self.configs_len)) + self.config_space._num_of_invalid_configs
            self.bohb_max_search_time_s = float(os.environ.get("TRITON_DEJAVU_BO_MAX_SEARCH_TIME", 'inf'))
            # self.bohb_max_search_time_s = int(os.environ.get("TRITON_DEJAVU_BO_MAX_SEARCH_TIME", 2147483647))
            # self.bohb_max_search_time_s = os.environ.get("TRITON_DEJAVU_BO_MAX_SEARCH_TIME", None)  # will be converted by library
            if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
                print(f"[triton-dejavu] Set n_trials for BOHB to {self.bohb_max_n_trials} and max walltime to {self.bohb_max_search_time_s}s (invalid configs in space: {self.config_space._num_of_invalid_configs}).")
        
        self._param_hash = self._get_param_hash()
        # self.cache = {}
        self.cache = global_dejavu_storage.restore_autotuner_cache(
            fn, self.configs_hash, self.key_hash, self._param_hash
        )
        if os.environ.get("TRITON_DEJAVU_USE_ONLY_RESTORED", "0") == "1":
            self.configs = global_dejavu_storage.get_used_configs(
                fn, self.configs_hash, self.key_hash, self._param_hash
            )
            # important, don't update configs_hash
            if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
                print(
                    f"[triton-dejavu] restricted configs for {str(fn)} to {len(self.configs)} used in the cache."
                )
        self.fallback_heuristic = fallback_heuristic

    def _get_param_hash(self):
        # hs = f"autotuner params: warmup {self.warmup_t} rep {self.rep_t} cuda_graphs {self.use_cuda_graph}"
        hs = f"autotuner params: warmup {self.warmup_t} rep {self.rep_t} cuda_graphs {self.use_cuda_graph} use_bo {self.use_bo} " \
             f"bo_max_trials {self.bohb_max_n_trials} bo_timeout {self.bohb_max_search_time_s}"
        # not relevant
        # hs += get_list_hash(self.reset_idx)
        # hs += get_list_hash(self.restore_idx)
        # TODO: how to hash the custom hooks?
        #  inspect cant find it, possible would be str(inspect.Signature().from_callable(self.pre_hook))
        #  maybe not relevant since should not influence the autotuner result
        h = get_string_hash(hs)
        return h

    def _bench(self, *args, config, **meta):
        if triton_major_version >= 3:
            from triton.compiler.errors import CompileTimeAssertionFailure

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        if triton_major_version >= 3:
            if not hasattr(config, "all_kwargs"):
                config.all_kwargs = lambda: _all_kwargs(config)
            current = dict(meta, **config.all_kwargs())
        else:
            current = dict(meta, **config.kwargs)
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            # print('finished pre_hook')
            if triton_major_version >= 3:
                self.fn.run(
                    *args,
                    # num_warps=config.num_warps,
                    # num_stages=config.num_stages,
                    # num_ctas=config.num_ctas,
                    **current,
                )
            else:
                self.fn.run(
                    *args,
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                    num_ctas=config.num_ctas,
                    enable_warp_specialization=config.enable_warp_specialization,
                    # enable_persistent=False,
                    **current,
                )
            self.post_hook(args)

        if triton_major_version >= 3:
            try:
                if self.use_cuda_graph:
                    # with torch.cuda.stream(self.benchmarking_stream):
                    #     bench_res = upstream_do_bench_cudagraph(kernel_call, rep=self.rep_t, return_mode="median")
                    kernel_call_obj = KernelEvalCall(self.fn, self.arg_names, self.benchmarking_stream, kernel_call, *args, **current)
                    use_isolated_process = False 
                    if os.environ.get('TRITON_DEJAVU_USE_ISOLATED_PROCESS', '0') == '1':
                        use_isolated_process = True
                    bench_res = do_bench_cudagraph(kernel_call_obj, rep=self.rep_t, return_mode='median', 
                                                   use_isolated_process=use_isolated_process, run_id=self.run_id, 
                                                   path_prefix=str(self._obj_hash))
                    return bench_res
                return upstream_do_bench(
                    kernel_call,
                    warmup=self.warmup_t,
                    rep=self.rep_t,
                    quantiles=(0.5, 0.2, 0.8),
                    fast_flush=False,
                )
            except (OutOfResources, CompileTimeAssertionFailure, RuntimeError) as e:
                if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
                    print(f"[triton-dejavu] testing config '{config}' failed with: '{e}'")
                # trying to avoid/reset CUDA error: an illegal memory access was encountered
                # gc.collect()
                # torch.cuda.empty_cache()
                # torch.cuda.ipc_collect()
                # traceback.clear_frames(sys.exc_info()[2])
                return (
                    float("inf")
                    if self.use_cuda_graph
                    else [float("inf"), float("inf"), float("inf")]
                )
            except AssertionError as e:
                print(f"ERROR: {e}")
                return (
                    float("inf")
                    if self.use_cuda_graph
                    else [float("inf"), float("inf"), float("inf")]
                )
        else:
            try:
                return upstream_do_bench(
                    kernel_call,
                    warmup=self.warmup_t,
                    rep=self.rep_t,
                    quantiles=(0.5, 0.2, 0.8),
                    fast_flush=False,
                )
            except OutOfResources:
                return [float("inf"), float("inf"), float("inf")]
            except AssertionError as e:
                print(f"ERROR: {e}")
                return [float("inf"), float("inf"), float("inf")]

    def _run_benchmarks(self, *args, configs, **kwargs):
        if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
            print(f"[triton-dejavu] Started benchmarking of {len(configs)} configurations... (use_bo: {self.use_bo})")
        if not self.use_bo:
            timings = {
                config: self._bench(*args, config=config, **kwargs)
                for config in configs
            }
            best_config = builtins.min(timings, key=timings.get)

        else:
            from ConfigSpace import Configuration as BohbConfiguration
            # from ConfigSpace import ConfigurationSpace as BohbConfigurationSpace
            import numpy as np
            from smac import HyperparameterOptimizationFacade, Scenario
            
            # config/silence loggers...
            # 10 = DEBUG, 30 = WARNING, 40 = ERROR
            logger_level = 40
            # doesn't work... (also not if done before the other imports)
            # from smac.utils.logging import setup_logging
            # setup_logging(logger_level)
            import smac
            smac.runner.abstract_runner.logger.setLevel(logger_level)
            smac.main.smbo.logger.setLevel(logger_level)
            smac.model.abstract_model.logger.setLevel(logger_level)
            smac.intensifier.abstract_intensifier.logger.setLevel(logger_level)
            smac.initial_design.abstract_initial_design.logger.setLevel(logger_level)
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning) 

            # # self._restore_args = {i:a.clone() if isinstance(a, torch.Tensor) else i:a for i, a in enumerate(args)}
            # self._restore_args = {i : a.clone() for i, a in enumerate(args) if isinstance(a, torch.Tensor)}

            def eval_config(config: BohbConfiguration, seed: int = 0) -> float:
                if not self.config_space.is_allowed_BohbConfig(config):
                    return float('nan')
                triton_config = self.config_space.convert_BohbConfig_to_Triton(config)
                print(f'\ntesting {triton_config}')
                # Necessary to avoid persistent RuntimeErrors
                # args_copy = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
                # bench_timings = self._bench(*args_copy, config=triton_config, **kwargs)
                bench_timings = self._bench(*args, config=triton_config, **kwargs)
                print(f'_bench returned {bench_timings}')
                if self.use_cuda_graph:
                    return bench_timings
                return bench_timings[0]
        

            smac_scenario = Scenario(self.bohb_config_space, deterministic=True, n_trials=self.bohb_max_n_trials, 
                                walltime_limit=self.bohb_max_search_time_s, n_workers=1)
            # print('starting smac...')
            smac_facade = HyperparameterOptimizationFacade(smac_scenario, eval_config, overwrite=True, dask_client=None)
            # need to force reset...
            # smac._optimizer._finished = False
            best_config_bohb = smac_facade.optimize()
            
            best_config = self.config_space.convert_BohbConfig_to_Triton(best_config_bohb)
            run_history = smac_facade.runhistory
            num_tested_configs = run_history.finished
            # [1, num_tested_configs], not 0 indexed!! 
            tested_configs = dict(run_history.ids_config)
            results_per_config = dict(run_history._cost_per_config)
            # print(results_per_config)
            # indexed with TrialKey...
            complete_data_per_config = dict(run_history._data)
            # list(complete_data_per_config.keys())
            total_smac_run_time = smac_facade.optimizer.used_walltime
            total_optimizer_time = smac_facade.optimizer.used_target_function_walltime
            failed_configs = [cid for cid, v in results_per_config.items() if (np.isnan(v) or np.isinf(v))]
            worked_configs = [cid for cid, v in results_per_config.items() if not (np.isnan(v) or np.isinf(v))]
            # print(failed_configs)
            # tested_configs = run_history.get_configs()
            # trials = [run_history.get_trials(c) for c in tested_configs]  #  list[TrialInfo]
            # result_trial_info = run_history.get_trials(best_config)[0]
            result_trial_info = run_history.get_trials(best_config_bohb)
            result_cost = run_history.get_cost(best_config_bohb)
            # print(result_trial_info)
            # print(result_cost)
            # result_trial_info_str = f"TrialInfo({result_trial_info.instance}, {result_trial_info.seed}, {result_trial_info.budget})"
            # timings = {best_config: result_trial_info_str}
            # timings = {best_config: [0.001]}
            timings = {best_config: result_cost}
            if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
                print(f"[triton-dejavu] BOHB finished after {total_smac_run_time}s (optimizer {total_optimizer_time}s), tested {num_tested_configs}, "
                      f"of which {len(failed_configs)} failed.")
                print(f"failed ids: {failed_configs}")
                print(f"worked ids: {worked_configs}")
            
            # for i, r in self._restore_args.items():
            #         args[i].copy_(r)

        self.run_id += 1
        return timings, best_config


    def run(self, *args, **kwargs):
        given_kwargs = list(kwargs.keys())
        required_config_args = self.config_kw_names + __additional_config_arg_check__
        if any(x in given_kwargs for x in required_config_args):
            if os.environ.get("TRITON_DEJAVU_DEBUG_DEBUG", "0") == "1":
                print(f"Triton autotuning skipped, using given config: {kwargs}.")
            ret = self.fn.run(
                *args,
                **kwargs,
            )
        else:
            self.nargs = dict(zip(self.arg_names, args))
            used_cached_result = True
            self.bench_time = 0.0
            if len(self.configs) > 1:
                all_args = {**self.nargs, **kwargs}
                _args = []
                for name in self.arg_names:
                    if name in all_args:
                        _args.append(all_args[name])
                key = [_args[i] for i in self.key_idx]
                for arg in _args:
                    if hasattr(arg, "dtype"):
                        key.append(str(arg.dtype))
                # to avoid encoding conflicts
                key_s = [str(k) for k in key]
                key_orig = key
                key = tuple(key_s)
                if key not in self.cache:
                    if os.environ.get('TRITON_DEJAVU_FORCE_FALLBACK', '0') == '0':
                        # prune configs
                        used_cached_result = False
                        pruned_configs = self.prune_configs(kwargs)
                        bench_start = time.time()
                        timings,  best_config = self._run_benchmarks(*args, configs=pruned_configs, **kwargs)
                        bench_end = time.time()
                        self.bench_time = bench_end - bench_start
                        self.cache[key] = best_config 
                        self._timings[key] = timings[self.cache[key]]
                        if (self.use_cuda_graph and self._timings[key] == float("inf")) or (
                            not self.use_cuda_graph and self._timings[key][0] == float("inf")
                        ):
                            raise RuntimeError(
                                f"All autotune examples failed (timing is {self._timings[key]})."
                            )
                        self.configs_timings = timings
                        self.pre_hook(args, reset_only=True)
                    else:
                        self.cache[key] = self.fallback_heuristic(key_orig)
                        print(f'[triton-dejavu] Determined config {self.cache[key]} based on heuristics for key {key_orig}.')
                config = self.cache[key]
            else:
                config = self.configs[0]
            self.best_config = config
            if not used_cached_result:
                global_dejavu_storage.add_autotuner_cache(
                    self.cache,
                    self.fn,
                    self.configs_hash,
                    self.key_hash,
                    self._param_hash,
                    self.configs_len,
                    self._timings,
                    self.rep_t,
                    self.warmup_t,
                    self.bench_time,
                )
                if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                    print(
                        f"Triton autotuning for function {self.base_fn.__name__} finished after "
                        f"{self.bench_time:.2f}s; best config selected: {self.best_config} with benchmark time {self._timings[key]}; "
                        f" evaluated {len(pruned_configs)} configurations;"
                    )
            full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
            if config.pre_hook is not None:
                config.pre_hook(full_nargs)
            if triton_major_version >= 3:
                if not hasattr(config, "all_kwargs"):
                    config.all_kwargs = lambda: _all_kwargs(config)
                ret = self.fn.run(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
                )
            else:
                ret = self.fn.run(
                    *args,
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                    num_ctas=config.num_ctas,
                    enable_warp_specialization=config.enable_warp_specialization,
                    **kwargs,
                    **config.kwargs,
                )
            self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                if triton_major_version >= 3:
                    for config in pruned_configs:
                        if not hasattr(config, "all_kwargs"):
                            config.all_kwargs = lambda: _all_kwargs(config)
                    est_timing = {
                        config: self.perf_model(
                            **self.nargs,
                            **kwargs,
                            **config.all_kwargs(),
                        )
                        for config in pruned_configs
                    }
                else:
                    est_timing = {
                        config: self.perf_model(
                            **self.nargs,
                            **kwargs,
                            **config.kwargs,
                            num_stages=config.num_stages,
                            num_warps=config.num_warps,
                            num_ctas=config.num_ctas,
                        )
                        for config in pruned_configs
                    }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[
                    :top_k
                ]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            if triton_major_version >= 3:
                if not hasattr(config, "all_kwargs"):
                    config.all_kwargs = lambda: _all_kwargs(config)
                ret.append(
                    self.fn.warmup(
                        *args,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                )
            else:
                ret.append(
                    self.fn.warmup(
                        *args,
                        num_warps=config.num_warps,
                        num_ctas=config.num_ctas,
                        num_stages=config.num_stages,
                        enable_warp_specialization=config.enable_warp_specialization,
                        enable_persistent=config.enable_persistent,
                        **kwargs,
                        **config.kwargs,
                    )
                )
        self.nargs = None
        return ret


def autotune(
    key,
    configs=None,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=25,
    rep=100,
    use_cuda_graph=False,
    config_space=None,
    use_bo=False,
    fallback_heuristic=None,
):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton_dejavu.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'args': a list of arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'args': a list of arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: Warmup time (in ms) to pass to benchmarking, defaults to 5.
    :type warmup: int
    :param rep: Repetition time (in ms) to pass to benchmarking, defaults to 50.
    :type rep: int
    :param config_space: The Configuration Space to generate configs from. Only one of configs or config_space can be set.
    :type config_space: triton_dejavu.ConfigSpace
    :param use_bo: Activate Bayesian Optimization (BO) to speed up autotuner runs (at the expense of allowing some percentage of performance drop of the choosen kernel). 
                   This feature can only be used in combination with config_space. Also, prune_configs_by must not be provided. 
    :type use_bo: bool
    :param fallback_heuristic: A lambda function to determine the used configuration in case `TRITON_DEJAVU_FORCE_FALLBACK=1` and no entry is found in the cache.
    :type fallback_heursitic: callable(key)
    """

    def decorator(fn):
        return Autotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            config_space,
            use_bo,
            fallback_heuristic,
        )

    return decorator


class ConfigSpace:
    """
    An object to represent the space of possible kernel configurations for the auto-tuner to evaluate.
    At the initalization of the autotuner, a list of all possible and valid configurations is generated
    and passed to the autotuner.

    example:
    .. highlight:: python
    .. code-block:: python

        @triton_dejavu.autotune(
            config_space=triton_dejavu.ConfigSpace(
                {'BLOCK_N_SIZE': [1024, 2048, 4096]},
                num_warps=[4, 8, 16],
                num_stages=[1, 2, 4, 6],
                num_ctas=[1]
            ),

    :ivar kwargs_with_lists: a dictionary of lists of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, List[Any]]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type num_ctas: int
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :type maxnreg: Optional[int]
    :ivar enable_warp_specialization: enable specialization (spatial partitioning) or not.
                                      See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#spatial-partitioning-also-known-as-warp-specialization (only triton < 3.0)
    :type enable_warp_specialization: bool
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    :ivar kwarg_conditions: a list of functions to be evaluated during configuration creation. The functions are called
                            with the generated kwarg dictionary. Only configuration combinations where all functions
                            evaluate to True are passed to the autotuner.
    """

    def __init__(
        self,
        kwargs_with_lists,
        num_warps=None,
        num_stages=None,
        num_ctas=None,
        enable_warp_specialization=None,
        pre_hook=None,
        kwarg_conditions=None,
    ):
        if num_warps is None:
            num_warps = [4]
        if num_stages is None:
            num_stages = [2]
        if num_ctas is None:
            num_ctas = [1]
        else:
            # check if other ctas are allowed
            import torch

            capability = torch.cuda.get_device_capability()
            if capability[0] < 9:
                num_ctas = [1]
        if enable_warp_specialization is None or triton_major_version >= 3:
            enable_warp_specialization = [False]
        if kwarg_conditions is None:
            kwarg_conditions = []
        self.kwargs = kwargs_with_lists
        self.kwarg_keys = list(kwargs_with_lists.keys())
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.enable_warp_specialization = enable_warp_specialization
        # self.enable_persistent = False
        self.pre_hook = pre_hook
        self.kwarg_conditions = kwarg_conditions
        self._num_of_invalid_configs = 0

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        if triton_major_version < 3:
            res.append(f"enable_warp_specialization: {self.enable_warp_specialization}")
        # res.append(f"enable_persistent: {self.enable_persistent}")
        return "ConfigSpace: " + ", ".join(res)

    def generate_config_list(self):
        # first generate cross product of kwargs
        ks = list(self.kwargs.keys())
        vs = list(self.kwargs.values())
        vs_product = list(itertools.product(*vs))
        kwarg_lists_complete = []
        for cur_combination in vs_product:
            nd = dict(zip(ks, cur_combination))
            kwarg_lists_complete.append(nd)
        # check for conditions
        kwarg_lists = []
        for kwarg in kwarg_lists_complete:
            append = True
            for condition in self.kwarg_conditions:
                # global AND
                if not condition(kwarg):
                    append = False
                    self._num_of_invalid_configs += 1
            if append:
                kwarg_lists.append(kwarg)
        # then cross product with all others
        if triton_major_version >= 3:
            config_product = list(
                itertools.product(self.num_warps, self.num_ctas, self.num_stages)
            )
        else:
            config_product = list(
                itertools.product(
                    self.num_warps,
                    self.num_ctas,
                    self.num_stages,
                    self.enable_warp_specialization,
                )
            )
        all_product = list(itertools.product(kwarg_lists, config_product))
        config_list = []
        for cc in all_product:
            if triton_major_version >= 3:
                nc = Config(
                    cc[0],
                    num_warps=cc[1][0],
                    num_ctas=cc[1][1],
                    num_stages=cc[1][2],
                    pre_hook=self.pre_hook,
                )
            else:
                nc = Config(
                    cc[0],
                    num_warps=cc[1][0],
                    num_ctas=cc[1][1],
                    num_stages=cc[1][2],
                    enable_warp_specialization=cc[1][3],
                    pre_hook=self.pre_hook,
                )
            config_list.append(nc)
        if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
            print(
                f"[triton-dejavu] generated {len(config_list)} configurations out of {str(self)}."
            )
        return config_list

    def get_BohbConfigSpace(self):
        from ConfigSpace import ConfigurationSpace as BohbConfigurationSpace
        config_space_dict = {}
        config_space_dict.update(self.kwargs)
        # TODO: make dynamic
        config_space_dict['num_warps'] = self.num_warps
        config_space_dict['num_stages'] = self.num_stages
        config_space_dict['num_ctas'] = self.num_ctas
        cs = BohbConfigurationSpace(config_space_dict)
        return cs

    def is_allowed_BohbConfig(self, bohb_config) -> bool:
        # kwarg = bohb_config
        kwarg = {}
        bohb_config_dict = dict(bohb_config)
        for k in self.kwarg_keys:
            kwarg[k] = bohb_config_dict[k]
        # print(kwarg)
        for i, condition in enumerate(self.kwarg_conditions):
            # global AND
            if not condition(kwarg):
                if os.environ.get("TRITON_DEJAVU_DEBUG_DEBUG", "0") == "1":
                    print(f'config {kwarg} is not allowed (violated condition {i})!')
                return False
        return True

    def convert_BohbConfig_to_Triton(self, bohb_config) -> Config:
        assert triton_major_version >= 3
        # nc = Config(pre_hook=self.pre_hook, **bohb_config)
        kwarg = {}
        bohb_config_dict = dict(bohb_config)
        for k in self.kwarg_keys:
            kwarg[k] = bohb_config_dict[k]
        nc = Config(
            kwarg, 
            # TODO: make dynamic
            num_warps=bohb_config_dict['num_warps'],
            num_ctas=bohb_config_dict['num_ctas'],
            num_stages=bohb_config_dict['num_stages'],
            pre_hook=self.pre_hook,
            )
        # print(nc)
        return nc
