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
import os
import time
import inspect
from typing import Dict
import itertools
import torch

from triton.testing import do_bench, do_bench_cudagraph
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
    get_triton_config_defaults,
    flag_print_debug,
    flag_print_autotuning,
    flag_print_debug_verbose,
)

if triton_major_version >= 3:
    from triton.compiler.errors import CompileTimeAssertionFailure
else:
    # to be backwards compatible
    class CompileTimeAssertionFailure(CompilationError):
        pass


__additional_config_arg_check__ = ["num_warps", "num_stages"]
__triton_config_parameter_names__ = get_triton_config_parameter_names()
__triton_config_default_values__ = get_triton_config_defaults()


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
        fallback_heuristic=None,
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

        self._param_hash = self._get_param_hash()
        self.cache = global_dejavu_storage.restore_autotuner_cache(
            fn, self.configs_hash, self.key_hash, self._param_hash
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
        self._use_fallback = os.environ.get("TRITON_DEJAVU_FORCE_FALLBACK", "0") == "1"

    def _get_param_hash(self):
        hs = f"autotuner params: warmup {self.warmup_t} rep {self.rep_t} cuda_graphs {self.use_cuda_graph}"
        # not relevant
        # hs += get_list_hash(self.reset_idx)
        # hs += get_list_hash(self.restore_idx)
        # TODO: how to hash the custom hooks?
        #  inspect cant find it, possible would be str(inspect.Signature().from_callable(self.pre_hook))
        #  maybe not relevant since should not influence the autotuner result
        h = get_string_hash(hs)
        return h

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
            self.fn.run(
                *args,
                **current,
            )
            self.post_hook(args)

        try:
            if self.use_cuda_graph:
                with torch.cuda.stream(self.benchmarking_stream):
                    bench_res = do_bench_cudagraph(
                        kernel_call, rep=self.rep_t, return_mode="median"
                    )
                return bench_res
            return do_bench(
                kernel_call,
                warmup=self.warmup_t,
                rep=self.rep_t,
                quantiles=(0.5, 0.2, 0.8),
                fast_flush=False,
            )
        except (
            OutOfResources,
            CompileTimeAssertionFailure,
        ):
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

    def run(self, *args, **kwargs):
        given_kwargs = list(kwargs.keys())
        required_config_args = self.config_kw_names + __additional_config_arg_check__
        if any(x in given_kwargs for x in required_config_args):
            if flag_print_debug_verbose:
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
                    if not self._use_fallback:
                        if flag_print_debug:
                            print(
                                f"[triton-dejavu] {key} not in cache, starting to tune..."
                            )
                        # prune configs
                        used_cached_result = False
                        pruned_configs = self.prune_configs(kwargs)
                        bench_start = time.time()
                        timings = {
                            config: self._bench(*args, config=config, **kwargs)
                            for config in pruned_configs
                        }
                        bench_end = time.time()
                        self.bench_time = bench_end - bench_start
                        self.cache[key] = builtins.min(timings, key=timings.get)
                        self._timings[key] = timings[self.cache[key]]
                        if (
                            self.use_cuda_graph and self._timings[key] == float("inf")
                        ) or (
                            not self.use_cuda_graph
                            and self._timings[key][0] == float("inf")
                        ):
                            raise RuntimeError(
                                f"All autotune examples failed (timing is {self._timings[key]})."
                            )
                        self.configs_timings = timings
                        self.pre_hook(args, reset_only=True)
                    else:
                        self.cache[key] = self.fallback_heuristic(key_orig)
                        if flag_print_autotuning:
                            print(
                                f"[triton-dejavu] Determined config {self.cache[key]} based on heuristics for key {key_orig}."
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
                        f"Triton autotuning for function {self.base_fn.__name__} finished after "
                        f"{self.bench_time:.2f}s; best config selected: {self.best_config} with benchmark time {self._timings[key]}; "
                        f" evaluated {len(pruned_configs)} configurations;"
                    )
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
        )

    return decorator


class ConfigSpace:
    """
    An object to represent the space of possible kernel configurations for the auto-tuner to evaluate.
    At the initalization of the autotuner, a list of all possible and valid configurations is generated
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
        self.pre_hook = pre_hook
        self.kwarg_conditions = kwarg_conditions

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
