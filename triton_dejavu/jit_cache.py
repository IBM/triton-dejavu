#  /*******************************************************************************
#   * Copyright 2025 IBM Corporation
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

import sys
import os
import time
import inspect
import copy

from triton import KernelInterface
from triton.runtime.driver import driver
from triton.runtime.autotuner import OutOfResources

from triton import __version__ as triton_version

triton_major_version = int(triton_version.split(".")[0])
triton_minor_version = int(triton_version.split(".")[1])
triton_version_float = triton_major_version + float(triton_minor_version / 10)

from triton_dejavu.dejavu_utilities import (
    flag_print_debug,
    flag_print_debug_verbose,
)

__print_name__ = "triton-dejavu"


class CacheLock:

    def __init__(self, id="unknown"):
        self.is_locked = False
        self.id = id

    def lock(self):
        self.is_locked = True
        if flag_print_debug_verbose:
            print(f"[{__print_name__}] JitCache lock '{self.id}' is LOCKED.")

    def unlock(self):
        self.is_locked = False
        if flag_print_debug_verbose:
            print(f"[{__print_name__}] JitCache lock '{self.id}' is UNLOCKED.")


global_cache_lock = CacheLock("global")


class PreparedKernel33:
    def __init__(
        self,
        grid_obj,
        grid_example,
        cache_launch_grid,
        kernel,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        update_only_arg_names,
        bound_args,
        autotuner_results_dict,
        cache_key,
        device,
    ):
        self.grid_obj = grid_obj
        self.grid_is_callable = callable(grid_obj)
        self.grid_size = len(
            grid_example
        )  # grid_example is always not callable, so we need both
        self.cache_launch_grid = cache_launch_grid
        self.concrete_grid = (None, None, None)
        if cache_launch_grid:
            grid_0 = grid_example[0]
            grid_1 = grid_example[1] if self.grid_size > 1 else 1
            grid_2 = grid_example[2] if self.grid_size > 2 else 1
            self.concrete_grid = (grid_0, grid_1, grid_2)
        self.kernel = kernel
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.autotuner_results_dict = autotuner_results_dict

        self.arg_list = []
        self.update_args_index = {}
        # We construct the list of arguments that are passed to the compiled
        # kernel beforehand. For the arguments that could change each time the
        # kernel is called, store a dummy value that will be set each time
        # __call__ is called. For the arguments that are labeled as assumed to
        # be constant, we skip this step and use the initial stored values.
        for i, arg_n in enumerate(bound_args.keys()):
            if arg_n in update_only_arg_names:
                self.update_args_index[arg_n] = i
                self.arg_list.append("dummy_value")
            else:
                self.arg_list.append(bound_args[arg_n])

        self.device = device
        self._init_handles()

        if flag_print_debug_verbose:
            print("arguments that will be updated:")
            print(self.update_args_index)
            print(f"grid is callable: {self.grid_is_callable}")
            # print("launch metadata")
            # print(self.launch_metadata)
            if cache_launch_grid:
                print(f"cached grid: {self.concrete_grid}")

        self.cache_key = cache_key

    def _init_handles(self):
        """
        more or less redo what CompiledKernel._init_handles is doing
        (c.f. triton/python/triton/runtime/compiler.py:379)
        """
        self.run = driver.active.launcher_cls(self.kernel.src, self.kernel.metadata)
        # check once and not again
        self.dev_max_shared = driver.active.utils.get_device_properties(self.device)[
            "max_shared_mem"
        ]
        if self.kernel.metadata.shared > self.dev_max_shared:
            raise OutOfResources(
                self.metadata.shared, self.dev_max_shared, "shared memory"
            )
        self.module, self.function, self.n_regs, self.n_spills = (
            driver.active.utils.load_binary(
                self.kernel.name,
                self.kernel.kernel,
                self.kernel.metadata.shared,
                self.device,
            )
        )
        if flag_print_debug_verbose:
            print(
                f"kernel initialized: {self.n_regs}, {self.n_spills}, {self.function}"
            )

    def __call__(self, *args, **kwargs):
        assert len(args) == 0

        for arg_n, idx in self.update_args_index.items():
            self.arg_list[idx] = kwargs[arg_n]

        if self.cache_launch_grid:
            grid_0, grid_1, grid_2 = self.concrete_grid
        else:
            if self.grid_is_callable:
                for arg_n, value in self.autotuner_results_dict.items():
                    kwargs[arg_n] = value
                grid = kwargs["grid"](kwargs)
            else:
                grid = kwargs["grid"]
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

        stream = driver.active.get_current_stream(self.device)

        return self.run(
            grid_0,
            grid_1,
            grid_2,
            stream,
            self.function,
            self.kernel.packed_metadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *self.arg_list,
        )

    def get_key(self):
        return self.cache_key


class PreparedKernel32:
    def __init__(
        self,
        grid_obj,
        grid_example,
        cache_launch_grid,
        kernel,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        non_const_arg_names,
        assume_const_vals_dict,
        update_only_arg_names,
        autotuner_results_dict,
        cache_key,
        device,
    ):
        self.grid_obj = grid_obj
        self.grid_is_callable = callable(grid_obj)
        self.grid_size = len(
            grid_example
        )  # grid_example is always not callable, so we need both
        self.cache_launch_grid = cache_launch_grid
        self.concrete_grid = (None, None, None)
        if cache_launch_grid:
            grid_0 = grid_example[0]
            grid_1 = grid_example[1] if self.grid_size > 1 else 1
            grid_2 = grid_example[2] if self.grid_size > 2 else 1
            self.concrete_grid = (grid_0, grid_1, grid_2)
        self.kernel = kernel
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.autotuner_results_dict = autotuner_results_dict

        self.non_const_arg_names = non_const_arg_names
        self.non_const_vals_lst = []
        self.update_args_index = {}
        # We construct the list of arguments that are passed to the compiled
        # kernel beforehand. For the arguments that could change each time the
        # kernel is called, store a dummy value that will be set each time
        # __call__ is called. For the arguments that are labeled as assumed to
        # be constant, we skip this step and use the initial stored values.
        for i, arg_n in enumerate(self.non_const_arg_names):
            if arg_n in update_only_arg_names:
                self.update_args_index[arg_n] = i
                self.non_const_vals_lst.append("dummy_value")
            else:
                self.non_const_vals_lst.append(assume_const_vals_dict[arg_n])

        self.device = device
        self._init_handles()

        if flag_print_debug_verbose:
            print("arguments that will be updated:")
            print(self.non_const_arg_names)
            print(f"grid is callable: {self.grid_is_callable}")
            # print("launch metadata")
            # print(self.launch_metadata)
            if cache_launch_grid:
                print(f"cached grid: {self.concrete_grid}")

        self.cache_key = cache_key

    def _init_handles(self):
        """
        more or less redo what CompiledKernel._init_handles is doing
        (c.f. triton/python/triton/runtime/compiler.py:379)
        """
        self.run = driver.active.launcher_cls(self.kernel.src, self.kernel.metadata)
        # check once and not again
        self.dev_max_shared = driver.active.utils.get_device_properties(self.device)[
            "max_shared_mem"
        ]
        if self.kernel.metadata.shared > self.dev_max_shared:
            raise OutOfResources(
                self.metadata.shared, self.dev_max_shared, "shared memory"
            )
        self.module, self.function, self.n_regs, self.n_spills = (
            driver.active.utils.load_binary(
                self.kernel.name,
                self.kernel.kernel,
                self.kernel.metadata.shared,
                self.device,
            )
        )
        if flag_print_debug_verbose:
            print(
                f"kernel initialized: {self.n_regs}, {self.n_spills}, {self.function}"
            )

    def __call__(self, *args, **kwargs):
        assert len(args) == 0

        for arg_n, idx in self.update_args_index.items():
            self.non_const_vals_lst[idx] = kwargs[arg_n]

        if self.cache_launch_grid:
            grid_0, grid_1, grid_2 = self.concrete_grid
        else:
            if self.grid_is_callable:
                for arg_n, value in self.autotuner_results_dict.items():
                    kwargs[arg_n] = value
                grid = kwargs["grid"](kwargs)
            else:
                grid = kwargs["grid"]
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

        stream = driver.active.get_current_stream(self.device)

        return self.run(
            grid_0,
            grid_1,
            grid_2,
            stream,
            self.function,
            self.kernel.packed_metadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *self.non_const_vals_lst,
        )

    def get_key(self):
        return self.cache_key


class JitCache(KernelInterface):

    def __init__(
        self,
        fn,
        arg_names,
        check_keys,
        check_specialization,
        cache_lock: CacheLock,
        cache_launch_grid=False,
        assume_const=None,
        autotuner_args=None,
    ):
        assert 3.0 <= triton_version_float <= 3.3
        if triton_version_float <= 3.2:
            self._get_prepared_kernel = self._get_prepared_kernel32
        else:
            self._get_prepared_kernel = self._get_prepared_kernel33

        # do first to support stacking of decorators
        self.arg_names = arg_names
        self.fn = fn

        if os.environ.get("TRITON_DEJAVU_DISABLE_JITCACHE", "0") == "1":
            # we are deactivated -> do nothing and set self.run
            #  to JitFunction.run
            self.run = fn.run
            return

        # if we have multiple decorators, the name is nested
        fnsl = str(fn).split(":")
        last_decorator = self
        last_fn = fn
        additional_decorators = []
        while len(fnsl) < 2:
            additional_decorators.append(str(last_fn).split(" ")[0].replace("<", ""))
            last_decorator = last_fn
            last_fn = fn.fn
            fnsl = str(last_fn).split(":")
        fn_name = fnsl[1][:-1]
        self._jit_fn = last_fn
        self._last_decorator_fn = last_decorator
        if flag_print_debug:
            print(
                f"[{__print_name__}] JITCache for Triton kernel {fn_name} is activated."
                f" Additional decorators: {additional_decorators}"
            )

        # self.base_fn = fn
        # while not inspect.isfunction(self.base_fn):
        #     self.base_fn = self.base_fn.fn
        self.cache_lock = cache_lock
        self.cache_launch_grid = cache_launch_grid
        self.dynamic_mode = False
        self.run = self._run_static
        if self.cache_lock is None:
            self.dynamic_mode = True
            self.run = self._run_dynamic
        self.check_keys = check_keys
        self.check_specialization = check_specialization
        self.assume_const = assume_const
        self.autotuner_args = []
        if autotuner_args is not None:
            self.autotuner_args = autotuner_args
        self.kernel_cache = {}

        def calc_cache_index(kwargs):
            cache_key = ""
            for c_arg_name in check_keys:
                cache_key += f"_{kwargs[c_arg_name]}"
            for c_arg_name in check_specialization:
                if kwargs[c_arg_name] == 1:
                    cache_key += f"__{c_arg_name} == 1__"
                elif kwargs[c_arg_name] % 16:
                    cache_key += f"__({c_arg_name} > 1) && ({c_arg_name} % 16 == 0)__"
                else:
                    cache_key += f"__({c_arg_name} > 1) && ({c_arg_name} % 16 != 0)__"
            return cache_key

        self.cache_index_func = calc_cache_index
        if len(check_keys) == 0 and len(check_specialization) == 0:
            self.cache_index_func = lambda ignore: "_default_"

    def _get_prepared_kernel33(self, *args, **kwargs) -> PreparedKernel33:
        """
        more or less redo what JITFunction.run is doing
        (c.f. triton/python/triton/runtime/jit.py:525)
        """

        kwargs["warmup"] = True
        compile_start = time.time()
        kernel = self.fn.run(*args, **kwargs)
        compile_end = time.time()

        const_arg_names = []
        non_const_arg_names = []
        for p in self._jit_fn.params:
            if p.name in self.autotuner_args:
                continue
            if p.is_constexpr or p.is_const:
                const_arg_names.append(p.name)
            else:
                non_const_arg_names.append(p.name)
        if any(x in self.check_keys for x in non_const_arg_names):
            raise RuntimeError(
                f"[{__print_name__}] ERROR: check_keys must only contain "
                "parameters marked as tl.constexpr (non-constants will be "
                "updated in all cases)."
            )
        if any(x in self.check_specialization for x in const_arg_names):
            raise RuntimeError(
                f"[{__print_name__}] ERROR: check_specialization must only contain "
                "integer parameters NOT marked as tl.constexpr."
            )
        if self.assume_const:
            if any(x in self.assume_const for x in const_arg_names):
                raise RuntimeError(
                    f"[{__print_name__}] ERROR: assume_const must only contain "
                    "parameters NOT marked as tl.constexpr."
                )
            update_only_arg_names = [
                arg_n for arg_n in non_const_arg_names if arg_n not in self.assume_const
            ]
        else:
            update_only_arg_names = non_const_arg_names

        const_arg_list = []
        for arg_n in const_arg_names:
            const_arg_list.append(kwargs[arg_n])

        autotuner_configs_dict = {}
        if len(self.autotuner_args) > 0:
            # if autotuner is triton_dejavu, we can determine the last missing arguments
            if hasattr(self._last_decorator_fn, "_last_complete_args"):
                for config_arg in self.autotuner_args:
                    kwargs[config_arg] = self._last_decorator_fn._last_complete_args[
                        config_arg
                    ]
                    autotuner_configs_dict[config_arg] = (
                        self._last_decorator_fn._last_complete_args[config_arg]
                    )
            else:
                raise RuntimeError(
                    f"[{__print_name__}] ERROR: cannot determine autotune results."
                )

        device = driver.active.get_current_device()
        kernel_cache, target, backend, binder = self._jit_fn.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)
        bind_end = time.time()

        if callable(kwargs["grid"]):
            grid = kwargs["grid"](kwargs)
        else:
            grid = kwargs["grid"]

        stream = driver.active.get_current_stream(device)
        launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())

        prepared_kernel = PreparedKernel33(
            kwargs["grid"],
            grid,
            self.cache_launch_grid,
            kernel,
            launch_metadata,
            self._jit_fn.CompiledKernel.launch_enter_hook,
            self._jit_fn.CompiledKernel.launch_exit_hook,
            update_only_arg_names,
            bound_args,
            autotuner_configs_dict,
            self.cache_index_func(kwargs),
            device,
        )

        wrapper_end = time.time()
        compile_time = compile_end - compile_start
        bind_time = bind_end - compile_end
        wrapper_time = wrapper_end - bind_end

        if flag_print_debug:
            print(
                f"[{__print_name__}] JIT compilation took {compile_time}s, binding {bind_time}, wrapper {wrapper_time}s."
            )

        return prepared_kernel

    def _get_prepared_kernel32(self, *args, **kwargs) -> PreparedKernel32:
        """
        more or less redo what JITFunction.run is doing
        (c.f. triton/python/triton/runtime/jit.py:565)
        """

        kwargs["warmup"] = True
        compile_start = time.time()
        kernel = self.fn.run(*args, **kwargs)
        compile_end = time.time()

        const_arg_names = []
        non_const_arg_names = []
        for p in self._jit_fn.params:
            if p.name in self.autotuner_args:
                continue
            if p.is_constexpr or p.is_const:
                const_arg_names.append(p.name)
            else:
                non_const_arg_names.append(p.name)
        if any(x in self.check_keys for x in non_const_arg_names):
            raise RuntimeError(
                f"[{__print_name__}] ERROR: check_keys must only contain "
                "parameters marked as tl.constexpr (non-constants will be "
                "updated in all cases)."
            )
        if any(x in self.check_specialization for x in const_arg_names):
            raise RuntimeError(
                f"[{__print_name__}] ERROR: check_specialization must only contain "
                "integer parameters NOT marked as tl.constexpr."
            )
        if self.assume_const:
            if any(x in self.assume_const for x in const_arg_names):
                raise RuntimeError(
                    f"[{__print_name__}] ERROR: assume_const must only contain "
                    "parameters NOT marked as tl.constexpr."
                )
            update_only_arg_names = [
                arg_n for arg_n in non_const_arg_names if arg_n not in self.assume_const
            ]
            assume_const_vals_dict = {
                arg_n: kwargs[arg_n]
                for arg_n in non_const_arg_names
                if arg_n in self.assume_const
            }
        else:
            update_only_arg_names = non_const_arg_names
            assume_const_vals_dict = {}

        autotuner_configs_dict = {}
        if len(self.autotuner_args) > 0:
            # if autotuner is triton_dejavu, we can determine the last missing arguments
            if hasattr(self._last_decorator_fn, "_last_complete_args"):
                for config_arg in self.autotuner_args:
                    kwargs[config_arg] = self._last_decorator_fn._last_complete_args[
                        config_arg
                    ]
                    autotuner_configs_dict[config_arg] = (
                        self._last_decorator_fn._last_complete_args[config_arg]
                    )
            else:
                raise RuntimeError(
                    f"[{__print_name__}] ERROR: cannot determine autotune results."
                )

        (
            bound_args,
            sig_and_spec,
            constexpr_vals,
            non_constexpr_vals,
            excess_kwargs,
        ) = self._jit_fn.binder(*args, **kwargs)
        bind_end = time.time()

        if callable(kwargs["grid"]):
            grid = kwargs["grid"](kwargs)
        else:
            grid = kwargs["grid"]

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)

        prepared_kernel = PreparedKernel32(
            kwargs["grid"],
            grid,
            self.cache_launch_grid,
            kernel,
            launch_metadata,
            self._jit_fn.CompiledKernel.launch_enter_hook,
            self._jit_fn.CompiledKernel.launch_exit_hook,
            non_const_arg_names,
            assume_const_vals_dict,
            update_only_arg_names,
            autotuner_configs_dict,
            self.cache_index_func(kwargs),
            device,
        )

        wrapper_end = time.time()
        compile_time = compile_end - compile_start
        bind_time = bind_end - compile_end
        wrapper_time = wrapper_end - bind_end

        if flag_print_debug:
            print(
                f"[{__print_name__}] JIT compilation took {compile_time}s, binding {bind_time}, wrapper {wrapper_time}s."
            )

        return prepared_kernel

    def _run_static(self, *args, **kwargs):
        # we only support kwargs
        if len(args) != 0:
            raise RuntimeError(
                f"[{__print_name__}] ERROR: The JITCache only supports kwargs, len(args) must be 0."
            )
        # assert no config pre-hook
        assert "pre_hook" not in kwargs or kwargs["pre_hook"] is None

        # print(f"my lock: {self.cache_lock.is_locked}")
        if not self.cache_lock.is_locked:
            # we only support int, bool, float as cache index
            for key in self.check_keys:
                if type(kwargs[key]) not in [int, bool, float, type(None)]:
                    raise RuntimeError(
                        f"[{__print_name__}] type of check_key {key} "
                        f"{type(kwargs[key])} is not one of supported types: "
                        f"int, bool float."
                    )
            # check_specialization must be int
            for key in self.check_specialization:
                if type(kwargs[key]) not in [int, type(None)]:
                    raise RuntimeError(
                        f"[{__print_name__}] type of check_specialization {key} "
                        f"{type(kwargs[key])} is not one of supported types: "
                        f"int."
                    )
            prepared_kernel = self._get_prepared_kernel(*args, **kwargs)
            if prepared_kernel.get_key() in self.kernel_cache and flag_print_debug:
                print(
                    f"[{__print_name__}:JitCache] WARNING: Kernel variant already cached, will override (cache lock is not locked). "
                    f"This could mean that the given check_keys are ambiguous (or the same call was already executed)."
                )
            self.kernel_cache[prepared_kernel.get_key()] = prepared_kernel

        try:
            kernel_variant = self.kernel_cache[self.cache_index_func(kwargs)]
        except KeyError as e:
            print(
                f"[{__print_name__}:JitCache] ERROR: Key {self.cache_index_func(kwargs)}  not in cache.\n"
                f"Current cache ({len(self.kernel_cache)} entries): {list(self.kernel_cache.keys())}"
            )
            print(e)
            raise e

        return kernel_variant(*args, **kwargs)

    def _run_dynamic(self, *args, **kwargs):
        # we only support kwargs
        if len(args) != 0:
            raise RuntimeError(
                f"[{__print_name__}] ERROR: The JITCache only supports kwargs, len(args) must be 0."
            )
        # assert no config pre-hook
        assert "pre_hook" not in kwargs or kwargs["pre_hook"] is None

        try:
            kernel_variant = self.kernel_cache[self.cache_index_func(kwargs)]
        except KeyError as e:
            if flag_print_debug:
                print(
                    f"[{__print_name__}:JitCache] Key {self.cache_index_func(kwargs)}  not in cache, compiling...\n"
                    f"Current cache ({len(self.kernel_cache)} entries): {list(self.kernel_cache.keys())}"
                )
            # we only support int, bool, float as cache index
            for key in self.check_keys:
                if type(kwargs[key]) not in [int, bool, float, type(None)]:
                    raise RuntimeError(
                        f"[{__print_name__}] type of check_key {key} "
                        f"{type(kwargs[key])} is not one of supported types: "
                        f"int, bool float."
                    ) from None
            # check_specialization must be int
            for key in self.check_specialization:
                if type(kwargs[key]) not in [int, type(None)]:
                    raise RuntimeError(
                        f"[{__print_name__}] type of check_specialization {key} "
                        f"{type(kwargs[key])} is not one of supported types: "
                        f"int."
                    ) from None
            kernel_variant = self._get_prepared_kernel(*args, **kwargs)
            self.kernel_cache[kernel_variant.get_key()] = kernel_variant

        return kernel_variant(*args, **kwargs)


def jitcache(
    check_keys,
    check_specialization,
    cache_lock=None,
    cache_launch_grid=False,
    assume_const=None,
    autotuner_args=None,
):
    """
    Decorator for caching a :code:`triton.jit`'d function.
    Basically, the :code:`JitCache` trades safety in all scenarios and high
    launch overhead of the original triton launcher against a low launch
    overhead but reduced/relaxed safety checks applicable only to applications-
    specific use. It is then the job of the developers to ensure that the
    relaxed safety checks still hold for the particular application.

    The :code:`JitCache` checks which compiled version of a kernel to use
    based on the mandatory :code:`check_keys` and :code:`check_specialization`
    lists. The developer needs to select these arguments based on her/his
    knowledge of the application.

    If a :code:`CacheLock` is provided, then the :code:`JitCache` adds new
    entries to the cache as long es the lock is unlocked. Once the CacheLock
    is locked and a kernel version is required that is not cached, it will
    throw an error.

    If no :code:`CacheLock` is provided, the :code:`JitCache` runs in the
    "dynamic" mode and creates new kernel variants if they are needed. This
    simplifies the application design but could add unexpected latency jitters.

    :param check_keys: The list of tl.constexpr that are used to index
                       the cache. Only types int, bool, float are supported.
    :type check_keys: list[str]
    :param check_specialization: The list of *non-constants* of type integer
                                 that are subject to specialization
                                 (e.g. values of the parameter could be 1 or
                                 could be dividable by 16).
    :type check_specialization: list[str]
    :param cache_lock: The CacheLock used for this JitCache.
    :type cache_lock: CacheLock
    :param cache_launch_grid: Indicate if the launch grid size is static and
                               should be cached (False by default).
    :type cache_launch_grid: bool
    :param assume_const: A list of parameters that are NOT marked as
                         tl.constexpr but should be treated as constants in
                         this kernel launch.
    :type assume_const: list[str]
    :param autotuner_args: A list of parameter names that are handled by an
                            autotuner decorator AFTER the jitcache.
    :type autotuner_args: list[str]
    """

    def decorator(fn):
        return JitCache(
            fn,
            fn.arg_names,
            check_keys,
            check_specialization,
            cache_lock,
            cache_launch_grid,
            assume_const,
            autotuner_args,
        )

    return decorator
