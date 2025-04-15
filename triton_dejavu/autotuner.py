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

from __future__ import annotations

import builtins
import sys
import os
import time
import inspect
from typing import Dict
import itertools
import torch
import copy

# TODO: still necessary?
# import gc
# import traceback

# from triton.testing import do_bench, do_bench_cudagraph
from triton_dejavu.testing import do_bench, KernelEvalCall
from triton import KernelInterface, Config, OutOfResources, CompilationError

from triton import __version__ as triton_version

triton_major_version = int(triton_version.split(".")[0])

from triton_dejavu.dejavu_storage import (
    global_dejavu_storage,
    get_config_list_hash,
    get_list_hash,
    get_string_hash,
)
from triton_dejavu.dejavu_utilities import (
    get_triton_config_parameter_names,
    get_triton_config_defaults_values,
    get_triton_config_defaults_types,
    flag_print_debug,
    flag_print_autotuning,
    flag_print_debug_verbose,
    get_type_dict,
)
from triton_dejavu.cache_manager import set_triton_cache_manager
from triton_dejavu.utils import global_metadata_store

if triton_major_version >= 3:
    from triton.compiler.errors import CompileTimeAssertionFailure
else:
    # to be backwards compatible
    class CompileTimeAssertionFailure(CompilationError):
        pass


__additional_config_arg_check__ = ["num_warps", "num_stages"]
__triton_config_parameter_names__ = get_triton_config_parameter_names()
__triton_config_default_values__ = get_triton_config_defaults_values()
__triton_config_default_types__ = get_triton_config_defaults_types()
__min_search_samples__ = int(os.getenv("_TRITON_DEJAVU_MIN_SEARCH_SAMPLES", "50"))


def _all_kwargs(self):
    """to be compatible with different triton versions"""
    for p in __triton_config_parameter_names__:
        if not hasattr(self, p):
            setattr(self, p, __triton_config_default_values__[p])
    return {
        **self.kwargs,
        **{
            p: getattr(self, p)
            for p in __triton_config_parameter_names__
            if getattr(self, p) is not None
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
        fallback_heuristic: callable = None,
        informed_fallback: callable = None,
        prepare_informed_fallback: callable = None,
        use_bo=False,
        use_random_search=False,
        search_max_search_t=180,
        search_max_share=1.0,
        search_max_repeat=1,
        quantiles=None,
        metadata_key=None,
        custom_data_storage=None,
    ):
        assert not (
            (informed_fallback is not None) and (fallback_heuristic is not None)
        ), "either fallback_heuristic or informed_fallback can be specified"
        assert not (
            use_bo and use_random_search
        ), "either use_bo or use_random_search can be set"
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
        # TODO: is pre-hook captured?
        #  no, it is not part of Config.__str__
        #  maybe not relevant?
        self.configs_hash = get_config_list_hash(self.configs)
        self.run_id = 0
        self._obj_hash = hash(self)
        # the key hash is not covered by fn.hash!
        self.key_hash = get_list_hash(key)
        self.orig_keys = key
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
        # self.quantiles = (0.5, 0.2, 0.8)
        self.quantiles = quantiles
        self.metadata_key = metadata_key

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
        self.use_random_search = use_random_search
        self.search_max_repeat = search_max_repeat
        self.search_max_n_trials = None
        self.max_search_time_s = search_max_search_t
        if self.use_bo:
            if not self.config_space or prune_configs_by:
                raise Exception(
                    f"[triton-dejavu] BO search can only be used in combination with config_space "
                    f"and without prune_configs_by!"
                )
            # doesn't work... (also not if done before the other imports)
            # from smac.utils.logging import setup_logging
            # setup_logging(40)
            # test imports
            from ConfigSpace import Configuration, ConfigurationSpace
            import numpy as np
            from smac import HyperparameterOptimizationFacade, Scenario

            self.search_max_n_trials = min(
                int(max(__min_search_samples__, search_max_share * self.configs_len))
                + self.config_space._num_of_invalid_configs,
                self.configs_len,
            )

            # convert config space
            self.bohb_config_space = self.config_space.get_BohbConfigSpace()
            if flag_print_debug:
                print(
                    f"[triton-dejavu] Set n_trials for BOHB to {self.search_max_n_trials} and max walltime to {self.max_search_time_s}s (invalid configs in space: {self.config_space._num_of_invalid_configs})."
                )
        if self.use_random_search:
            if prune_configs_by:
                raise Exception(
                    f"[triton-dejavu] Random search can only be used without prune_configs_by!"
                )
            # just test import, random list is generated at every search
            import numpy as np

            self.search_max_n_trials = min(
                int(max(__min_search_samples__, search_max_share * self.configs_len)),
                self.configs_len,
            )

            if flag_print_debug:
                print(
                    f"[triton-dejavu] Set n_trials for Random Search to {self.search_max_n_trials} and max walltime to {self.max_search_time_s}s (invalid configs in space: {self.config_space._num_of_invalid_configs})."
                )

        self._param_hash = self._get_param_hash()
        all_pre_hook = (
            self.config_space.get_global_pre_hook()
            if self.config_space is not None
            else None
        )
        if custom_data_storage:
            global_dejavu_storage.add_cache_data_path_prefix(
                custom_data_storage,
                fn,
                self.configs_hash,
                self.key_hash,
                self._param_hash,
            )
        self.cache = global_dejavu_storage.restore_autotuner_cache(
            fn,
            self.configs_hash,
            self.key_hash,
            self._param_hash,
            all_pre_hook=all_pre_hook,
        )
        if configs and len(self.cache) > 1:
            # iterate over given config list to detect pre_hooks on individual config level
            # pre_hooks of individual Configs are not part of the config-list hash
            #  (because it is not part of Config.__str__ but also, it shouldn't influence the autotuner result)
            for kt, config in self.cache.items():
                for oc in configs:
                    if str(oc) != str(config):
                        continue
                    if oc.pre_hook is not None:
                        config.pre_hook = oc.pre_hook
                        if flag_print_debug_verbose:
                            print(
                                f"[triton-dejavu] added pre_hook to restored config {config}."
                            )
                        self.cache[kt] = config

        if os.environ.get("TRITON_DEJAVU_USE_ONLY_RESTORED", "0") == "1":
            self.configs = global_dejavu_storage.get_used_configs(
                fn, self.configs_hash, self.key_hash, self._param_hash
            )
            # important, don't update configs_hash
            if flag_print_debug:
                print(
                    f"[triton-dejavu] restricted configs for {str(fn)} to {len(self.configs)} used in the cache."
                )

        self.fallback_heuristic = fallback_heuristic
        self.informed_fallback = informed_fallback
        self._cache_for_fallback = None
        if self.informed_fallback is not None:
            # we make a copy of the cache as it is after init, because the fallbacks will modify it themselves
            self._cache_for_fallback = copy.deepcopy(self.cache)
            if (
                prepare_informed_fallback is not None
                and len(self._cache_for_fallback) > 0
            ):
                self._cache_for_fallback = prepare_informed_fallback(
                    self._cache_for_fallback
                )
                assert (
                    self._cache_for_fallback
                ), "`prepare_informed_fallback must not return None"
                if flag_print_debug_verbose:
                    print(
                        f"prepared cache for informed fallback: {self._cache_for_fallback}"
                    )
        if prepare_informed_fallback is not None and informed_fallback is None:
            print(
                "[triton-dejavu] WARNING: prepare_informed_fallback will be ignored because informed_fallback is not specified."
            )
        if informed_fallback is not None and len(self._cache_for_fallback) == 0:
            print(
                "[triton-dejavu] WARNING: informed_fallback and prepare_informed_fallback will be ignored because existing cache is empty."
            )
        self._use_fallback = os.environ.get("TRITON_DEJAVU_FORCE_FALLBACK", "0") == "1"
        if self._use_fallback:
            assert (
                self.fallback_heuristic is not None
                or self.informed_fallback is not None
            ), "force to use fallback functions, but none specified"
            if self.informed_fallback is not None and len(self._cache_for_fallback) > 0:
                self._fallback_call = lambda key: self.informed_fallback(
                    key, self._cache_for_fallback
                )
                if flag_print_debug:
                    print(
                        f"[triton-dejavu] Using informed fallback function (custom cache prepare: {prepare_informed_fallback is not None})."
                    )
            else:
                self._fallback_call = self.fallback_heuristic
                if flag_print_debug:
                    print(
                        "[triton-dejavu] Using un-informed fallback heuristic function."
                    )

        self._use_isolated_process = (
            os.environ.get("TRITON_DEJAVU_USE_ISOLATED_PROCESS", "0") == "1"
        )

        # triton cache
        self._use_split_cache = os.environ.get("TRITON_DEJAVU_SPLIT_CACHE", "0") == "1"
        if self._use_split_cache:
            if flag_print_debug:
                print(f"[triton-dejavu] Triton cache isolation activated.")
            self._update_triton_cache_path()
            set_triton_cache_manager()
        self._start_time = time.time()

    def _get_param_hash(self):
        hs = f"autotuner params: warmup {self.warmup_t} rep {self.rep_t} cuda_graphs {self.use_cuda_graph} quantiles {self.quantiles}"
        # search params are not always managed by a tag, so per default we should hash them
        if os.getenv("TRITON_DEJAVU_HASH_SEARCH_PARAMS", "1") == "1":
            hs += f" use_bo {self.use_bo} use_random {self.use_random_search} max_search_n {self.search_max_n_trials} max_search_t {self.max_search_time_s}"
        # TODO: how to hash the custom hooks?
        #  inspect can't find it, possible would be str(inspect.Signature().from_callable(self.pre_hook))
        #  maybe not relevant since should not influence the autotuner result
        h = get_string_hash(hs)
        return h

    def _update_triton_cache_path(self):
        run_id_str = f"{self._obj_hash}-{self.run_id}"
        os.environ["TRITON_DEJAVU_INSTANCE_RUN_ID"] = run_id_str

    def _bench(self, *args, config, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        if not hasattr(config, "all_kwargs"):
            config.all_kwargs = lambda: _all_kwargs(config)
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            # print('finished pre_hook')
            self.fn.run(
                *args,
                **current,
            )
            self.post_hook(args)

        try:
            kernel_call_obj = KernelEvalCall(
                self.fn,
                self.arg_names,
                self.benchmarking_stream,
                config,
                kernel_call,
                *args,
                **current,
            )
            # NOTE: if a config.pre_hook exists, it will be executed in the parent process
            #  but with already cloned args (not possible to pickle an unbound kernel)
            bench_res = do_bench(
                kernel_call_obj,
                use_cuda_graphs=self.use_cuda_graph,
                warmup=self.warmup_t,  # for eager mode
                rep=self.rep_t,
                quantiles=self.quantiles,
                return_mode="median",
                use_isolated_process=self._use_isolated_process,
                run_id=self.run_id,
                path_prefix=str(self._obj_hash),
            )
            return bench_res
        except (
            OutOfResources,
            CompileTimeAssertionFailure,
            RuntimeError,
        ) as e:
            if flag_print_debug:
                print(f"[triton-dejavu] testing config '{config}' failed with: '{e}'")
            # trying to avoid/reset CUDA error: an illegal memory access was encountered
            # gc.collect()
            # torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()
            # traceback.clear_frames(sys.exc_info()[2])
            return (
                float("inf")
                # if self.use_cuda_graph
                if self.quantiles is None
                else [float("inf"), float("inf"), float("inf")]
            )
        except AssertionError as e:
            print(f"ERROR: {e}")
            return (
                float("inf")
                # if self.use_cuda_graph
                if self.quantiles is None
                else [float("inf"), float("inf"), float("inf")]
            )

    def _run_benchmarks(self, *args, configs, **kwargs):
        if self._use_split_cache:
            self._update_triton_cache_path()
        if flag_print_debug:
            print(
                f"[triton-dejavu] [{time.strftime('%Y-%m-%d %H:%M:%S')}]  Started benchmarking of {len(configs)} configurations... (use_bo: {self.use_bo}, run: {self.run_id})"
            )
        if not self.use_bo and not self.use_random_search:
            timings = {
                config: self._bench(*args, config=config, **kwargs)
                for config in configs
            }
            best_config = builtins.min(timings, key=timings.get)

        elif self.use_bo:
            from ConfigSpace import Configuration as BohbConfiguration

            # from ConfigSpace import ConfigurationSpace as BohbConfigurationSpace
            import numpy as np
            from smac import HyperparameterOptimizationFacade, Scenario
            from smac.acquisition.function.expected_improvement import EI

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
                    return float("nan")
                triton_config = self.config_space.convert_BohbConfig_to_Triton(config)
                print(f"\ntesting {triton_config}")
                # Necessary to avoid persistent RuntimeErrors
                # args_copy = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
                # bench_timings = self._bench(*args_copy, config=triton_config, **kwargs)
                bench_timings = self._bench(*args, config=triton_config, **kwargs)
                print(f"_bench returned {bench_timings}")
                # if self.use_cuda_graph:
                if self.quantiles is None:
                    return bench_timings
                return bench_timings[0]

            # TODO
            result_cost = float("inf")
            total_trials = 0
            n_trials = self.search_max_n_trials
            walltime_limit = self.max_search_time_s
            overwrite = True
            while np.isinf(result_cost) and total_trials < self.search_max_repeat:
                if total_trials > 0:
                    n_trials += self.search_max_n_trials
                    walltime_limit += self.max_search_time_s
                    overwrite = False
                    if flag_print_debug:
                        print(
                            f"[triton-dejavu] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Re-run BO search because all previous trials failed (total iteration :{total_trials})."
                        )

                smac_scenario = Scenario(
                    self.bohb_config_space,
                    deterministic=True,
                    n_trials=n_trials,
                    walltime_limit=walltime_limit,
                    n_workers=1,
                )
                # print('starting smac...')
                exploration_EI = EI(xi=0.04)
                smac_facade = HyperparameterOptimizationFacade(
                    smac_scenario,
                    eval_config,
                    overwrite=overwrite,
                    dask_client=None,
                    # acquisition_function=exploration_EI
                )
                # need to force reset...
                # smac._optimizer._finished = False

                best_config_bohb = smac_facade.optimize()

                best_config = self.config_space.convert_BohbConfig_to_Triton(
                    best_config_bohb
                )
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
                total_optimizer_time = (
                    smac_facade.optimizer.used_target_function_walltime
                )
                failed_configs = [
                    cid
                    for cid, v in results_per_config.items()
                    if (np.isnan(v) or np.isinf(v))
                ]
                worked_configs = [
                    cid
                    for cid, v in results_per_config.items()
                    if not (np.isnan(v) or np.isinf(v))
                ]
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
                if flag_print_debug:
                    print(
                        f"[triton-dejavu] [{time.strftime('%Y-%m-%d %H:%M:%S')}] BOHB finished after {total_smac_run_time}s (optimizer {total_optimizer_time}s), tested {num_tested_configs}, "
                        f"of which {len(failed_configs)} failed."
                    )
                    print(f"failed ids: {failed_configs}")
                    print(f"worked ids: {worked_configs}")
                total_trials += 1
            # for i, r in self._restore_args.items():
            #         args[i].copy_(r)

        elif self.use_random_search:
            import numpy as np

            rng = np.random.default_rng()
            start_time = time.time()
            total_trials = 0
            result_cost = float("inf")
            n_trials = self.search_max_n_trials
            walltime_limit = self.max_search_time_s
            timings = {}
            best_config = None
            num_tested_configs = 0
            failed_configs = []
            worked_configs = []
            while np.isinf(result_cost) and total_trials < self.search_max_repeat:
                if total_trials > 0:
                    n_trials += self.search_max_n_trials
                    walltime_limit += self.max_search_time_s
                    if flag_print_debug:
                        print(
                            f"[triton-dejavu] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Re-run random search because all previous trials failed (total iteration :{total_trials})."
                        )

                random_search_list = rng.choice(
                    len(self.configs), self.search_max_n_trials, replace=False
                )
                for ci in random_search_list:
                    this_config = self.configs[ci]
                    print(f"\ntesting {this_config}")
                    bench_timings = self._bench(*args, config=this_config, **kwargs)
                    print(f"_bench returned {bench_timings}")
                    timings[this_config] = bench_timings
                    if self.quantiles is None:
                        if bench_timings < result_cost:
                            best_config = this_config
                            result_cost = bench_timings
                        if np.isnan(bench_timings):
                            failed_configs.append(ci)
                        else:
                            worked_configs.append(ci)
                    else:
                        if bench_timings[0] < result_cost:
                            best_config = this_config
                            result_cost = bench_timings[0]
                        if np.isnan(bench_timings[0]):
                            failed_configs.append(ci)
                        else:
                            worked_configs.append(ci)
                    num_tested_configs += 1
                    if (start_time + walltime_limit) < time.time():
                        # timeout
                        print(f"random search timeout after {time.time() - start_time}")
                        break
                if flag_print_debug:
                    print(
                        f"[triton-dejavu] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Random Search finished after {time.time() - start_time}s, tested {num_tested_configs}, "
                        f"of which {len(failed_configs)} failed."
                    )
                    print(f"failed ids: {failed_configs}")
                    print(f"worked ids: {worked_configs}")
                total_trials += 1

        self.run_id += 1
        return timings, best_config

    def run(self, *args, **kwargs):
        given_kwargs = list(kwargs.keys())
        required_config_args = self.config_kw_names + __additional_config_arg_check__
        if any(x in given_kwargs for x in required_config_args):
            if flag_print_debug:
                print(
                    f"[triton-dejavu] Autotuning skipped, use config given as part of kwargs: {kwargs}."
                )
            # TODO: call pre_hook or kwargs['pre_hook']?
            if "pre_hook" in kwargs and kwargs["pre_hook"] is not None:
                nargs = dict(zip(self.arg_names, args))
                full_args = {**nargs, **kwargs}
                kwargs["pre_hook"](full_args)
                # to avoid KeyError: 'Keyword argument pre_hook was specified but unrecognised'
                del kwargs["pre_hook"]
            ret = self.fn.run(
                *args,
                **kwargs,
            )
        else:
            # FIXME: this could assign the wrong argument to the wrong name if autotuner args are not last!
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
                    # TODO: find better solution
                    # should_be_ready = (time.time() - self._start_time) > (5 * 60)
                    # if os.environ.get("TRITON_DEJAVU_FORCE_FALLBACK", "0") == "0" and should_be_ready:
                    if not self._use_fallback:
                        if flag_print_debug:
                            print(
                                f"[triton-dejavu] {key} not in cache, starting to tune..."
                            )
                        # prune configs
                        used_cached_result = False
                        pruned_configs = self.prune_configs(kwargs)
                        bench_start = time.time()
                        timings, best_config = self._run_benchmarks(
                            *args, configs=pruned_configs, **kwargs
                        )
                        bench_end = time.time()
                        self.bench_time = bench_end - bench_start
                        self.cache[key] = best_config
                        self._timings[key] = timings[self.cache[key]]
                        # if (
                        #     self.use_cuda_graph and self._timings[key] == float("inf")
                        # ) or (
                        #     not self.use_cuda_graph
                        #     and self._timings[key][0] == float("inf")
                        # ):
                        #     raise RuntimeError(
                        #         f"All autotune examples failed (timing is {self._timings[key]})."
                        #     )
                        if (
                            self.quantiles is None
                            and self._timings[key] == float("inf")
                        ) or (
                            not self.quantiles is None
                            and self._timings[key][0] == float("inf")
                        ):
                            raise RuntimeError(
                                f"All autotune examples failed (timing is {self._timings[key]})."
                            )
                        self.configs_timings = timings
                        self.pre_hook(args, reset_only=True)
                    else:
                        self.cache[key] = self._fallback_call(key_orig)
                        if flag_print_autotuning:
                            print(
                                f"[triton-dejavu] Determined config {self.cache[key]} based on heuristics for key {key_orig}."
                            )
                elif flag_print_debug_verbose:
                    print(
                        f"[triton-dejavu] Found config: {self.cache[key]} for key {key_orig}."
                    )
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
                    self.use_cuda_graph,
                    self.orig_keys,
                )
                if flag_print_autotuning:
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Triton autotuning for function {self.base_fn.__name__} finished after "
                        f"{self.bench_time:.2f}s; best config selected: {self.best_config} with benchmark time {self._timings[key]}; "
                        f" evaluated {len(pruned_configs)} configurations;"
                    )
            if self.metadata_key:
                global_metadata_store[self.metadata_key] = (
                    f"<autotune:{self.best_config}>"
                ).replace(" ", "")
            full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
            if config.pre_hook is not None:
                config.pre_hook(full_nargs)
            if not hasattr(config, "all_kwargs"):
                config.all_kwargs = lambda: _all_kwargs(config)
            ret = self.fn.run(
                *args,
                **kwargs,
                **config.all_kwargs(),
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
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[
                    :top_k
                ]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            if not hasattr(config, "all_kwargs"):
                config.all_kwargs = lambda: _all_kwargs(config)
            ret.append(
                self.fn.warmup(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
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
    fallback_heuristic=None,
    informed_fallback=None,
    prepare_informed_fallback=None,
    use_bo=False,
    use_random_search=False,
    search_max_search_t=180,
    search_max_share=1.0,
    search_max_repeat=1,
    quantiles=None,
    metadata_key=None,
    custom_data_storage=None,
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
    :param fallback_heuristic: A lambda function to determine the used configuration in case `TRITON_DEJAVU_FORCE_FALLBACK=1` and no entry is found in the cache.
    :type fallback_heursitic: callable(key)
    :param informed_fallback: A lambda function to determine the used configuration in case `TRITON_DEJAVU_FORCE_FALLBACK=1` and no entry is found in the cache.
                              This heuristic gets the cache as 2nd argument to make an *informed* decision based on the existing best known configs at start time.
                              If `prepare_informed_fallback` is defined, then the returned dict of this function will be provided.
    :type informed_fallback: callable(key, cache)
    :param prepare_informed_fallback: A lambda function to apply preprocessing to the existing autotuner cache at start time to facilitate the `informed_fallback`
                                      heuristic. The argument is the cache dict and any dict in return is expected.
    :type prepare_informed_fallback: callable(cache) -> dict
    :param use_bo: Activate Bayesian Optimization (BO) to speed up autotuner runs (at the expense of allowing some percentage of performance drop of the chosen kernel).
                   This feature can only be used in combination with config_space. Also, prune_configs_by must not be provided.
    :type use_bo: bool
    :param use_random_search: Activate Random Search to speed up autotuner runs (at the expense of allowing some percentage of performance drop of the chosen kernel).
                   This feature can be used in combination with config_space and config lists. However, prune_configs_by must not be provided.
    :type use_random_search: bool
    :param search_max_search_t: Maximum search time (in seconds) for BO and Random Search.
    :type search_max_search_t: int
    :param search_max_share: Maximum percentage of the total config space BO and Random Search can search through. This translates into a maximum trial number for the optimizer.
    :type search_max_share: float
    :param search_max_repeat: Maximum repetition of BO or Random Search in case all experiments failed (default 1). This is helpful if very large search spaces with
                              many invalid configurations are searched and therefore there exists a chance of finding only invalid configurations during a short search time.
    :type search_max_repeat: int
    :param quantiles: 3-tuple for the quantiles that are reported of the evaluation function, e.g. (0.5, 0.2, 0.8).
                        Default is `None` which will lead to the median (0.5 quantile).
    :param metadata_key: String to store the found configuration as metadata in the triton_dejavu.utils.global_metadata_store.
                         This could be used in combination with metadata_fn and proton.
    :type metadata_key: str
    :param custom_data_storage: Absolute path to a custom triton-dejavu data location for this function.
    :type custom_data_storage: str
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
            fallback_heuristic,
            informed_fallback,
            prepare_informed_fallback,
            use_bo,
            use_random_search,
            search_max_search_t,
            search_max_share,
            search_max_repeat,
            quantiles,
            metadata_key,
            custom_data_storage,
        )

    return decorator


class ConfigSpace:
    """
    An object to represent the space of possible kernel configurations for the auto-tuner to evaluate.
    At the initialization of the autotuner, a list of all possible and valid configurations is generated
    and passed to the autotuner.

    Please note that some of the configuration parameters depend on the used triton config.
    ConfigSpace will dynamically use the one of the installed triton platform.

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
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    :ivar kwarg_conditions: a list of functions to be evaluated during configuration creation. The functions are called
                            with the generated kwarg dictionary. Only configuration combinations where all functions
                            evaluate to True are passed to the autotuner.
    :ivar configuration_args: keyword arguments (so name=value, ...) for triton.Config parameters. For example, this are usually num_warps,
                              num_stages, num_ctas. Depending on version or platform, such as enable_warp_specialization
                              or maxnreg will be used as well.
    """

    def __init__(
        self,
        kwargs_with_lists,
        kwarg_conditions=None,
        pre_hook=None,
        **configuration_args,
    ):
        if kwarg_conditions is None:
            kwarg_conditions = []
        self.kwargs = kwargs_with_lists
        self.kwarg_keys = list(kwargs_with_lists.keys())
        self.kwarg_types = get_type_dict(
            {k: v[0] for k, v in kwargs_with_lists.items()}
        )
        self.pre_hook = pre_hook
        self.kwarg_conditions = kwarg_conditions
        self._num_of_invalid_configs = 0

        # adapt to current triton platform
        for k, v in __triton_config_default_values__.items():
            # but as lists!
            setattr(self, k, [v])

        for k, v in dict(configuration_args).items():
            if k not in __triton_config_parameter_names__:
                print(
                    f"[triton-dejav] WARNING: Configuration parameter {k} not supported on this platform and will be ignored."
                )
                continue
            setattr(self, k, v)

        # check for special parameters
        if hasattr(self, "num_ctas"):
            # check if other ctas are allowed
            import torch

            capability = torch.cuda.get_device_capability()
            if capability[0] < 9:
                self.num_ctas = [1]

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        for p in __triton_config_parameter_names__:
            res.append(f"{p}: {getattr(self, p)}")
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
        list_of_list_of_config_params = [
            getattr(self, p) for p in __triton_config_parameter_names__
        ]
        config_product = list(itertools.product(*list_of_list_of_config_params))
        all_product = list(itertools.product(kwarg_lists, config_product))
        config_list = []
        for cc in all_product:
            config_params = {}
            for i, p in enumerate(__triton_config_parameter_names__):
                config_params[p] = cc[1][i]
            nc = Config(
                cc[0],
                pre_hook=self.pre_hook,
                **config_params,
            )
            config_list.append(nc)
        if flag_print_debug:
            print(
                f"[triton-dejavu] generated {len(config_list)} configurations out of {str(self)}."
            )
        return config_list

    def get_BohbConfigSpace(self):
        from ConfigSpace import ConfigurationSpace as BohbConfigurationSpace

        config_space_dict = {}
        config_space_dict.update(self.kwargs)
        # TODO: make dynamic
        # config_space_dict["num_warps"] = self.num_warps
        # config_space_dict["num_stages"] = self.num_stages
        # config_space_dict["num_ctas"] = self.num_ctas
        for p in __triton_config_parameter_names__:
            config_space_dict[p] = getattr(self, p)
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
                if flag_print_debug_verbose:
                    print(f"config {kwarg} is not allowed (violated condition {i})!")
                return False
        return True

    def convert_BohbConfig_to_Triton(self, bohb_config) -> Config:
        assert triton_major_version >= 3
        # nc = Config(pre_hook=self.pre_hook, **bohb_config)
        kwarg = {}
        bohb_config_dict = dict(bohb_config)
        for k in self.kwarg_keys:
            kwarg[k] = self.kwarg_types[k](bohb_config_dict[k])
        config_params = {
            p: __triton_config_default_types__[p](bohb_config_dict[p])
            for p in __triton_config_parameter_names__
        }
        nc = Config(
            kwarg,
            # num_warps=int(bohb_config_dict["num_warps"]),
            # num_ctas=int(bohb_config_dict["num_ctas"]),
            # num_stages=int(bohb_config_dict["num_stages"]),
            pre_hook=self.pre_hook,
            **config_params,
        )
        # print(nc)
        return nc

    def get_global_pre_hook(self):
        return self.pre_hook
