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

from triton import KernelInterface
from triton.runtime.driver import driver

from triton import __version__ as triton_version

triton_major_version = int(triton_version.split(".")[0])
triton_minor_version = int(triton_version.split(".")[1])
triton_version_float = triton_major_version + float(triton_minor_version / 10)

from triton_dejavu.dejavu_utilities import (
    flag_print_debug,
    flag_print_debug_verbose,
)


class CacheLock:

    def __init__(self, id="unkown"):
        self.is_locked = False
        self.id = id

    def lock(self):
        self.is_locked = True
        if flag_print_debug_verbose:
            print(f"[triton-dejavu] JitCache lock '{self.id}' is LOCKED.")

    def unlock(self):
        self.is_locked = False
        if flag_print_debug_verbose:
            print(f"[triton-dejavu] JitCache lock '{self.id}' is UNLOCKED.")


global_cache_lock = CacheLock("global")


class PreparedKernel:
    def __init__(
        self,
        grid,
        kernel,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        non_const_arg_names,
        cache_key,
        device,
        stream,
    ):
        self.grid_obj = grid
        self.kernel = kernel
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.non_const_arg_names = non_const_arg_names

        self.cache_key = cache_key
        # TODO: safe to cache?
        self.device = device
        self.stream = stream

    def __call__(self, *args, **kwargs):
        assert len(args) == 0

        non_constsexpr_vals = []
        # order is always the same...
        for arg_n in self.non_const_arg_names:
            non_constsexpr_vals.append(kwargs[arg_n])

        if callable(self.grid_obj):
            grid = self.grid_obj(kwargs)
        else:
            grid = self.grid_obj
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1

        return self.kernel.run(
            grid_0,
            grid_1,
            grid_2,
            self.stream,
            self.kernel.function,
            self.kernel.packed_metadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *non_constsexpr_vals,
        )

    def get_key(self):
        return self.cache_key


class JitCache(KernelInterface):

    def __init__(
        self,
        fn,
        arg_names,
        cache_lock: CacheLock,
        check_keys,
    ):
        assert 3.0 <= triton_version_float <= 3.2
        self.arg_names = arg_names
        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self.cache_lock = cache_lock
        self.check_keys = check_keys
        self.kernel_cache = {}

        def calc_cache_index(kwargs):
            cache_key = ""
            for c_arg_name in check_keys:
                cache_key += str(kwargs[c_arg_name])
            return cache_key

        self.cache_index_func = calc_cache_index

    def _get_prepared_kernel(self, *args, **kwargs) -> PreparedKernel:

        kwargs["warmup"] = True
        compile_start = time.time()
        kernel = self.fn.run(*args, **kwargs)
        compile_end = time.time()

        const_arg_names = []
        non_const_arg_names = []
        for p in self.fn.params:
            if p.is_constexpr or p.is_const:
                const_arg_names.append(p.name)
            else:
                non_const_arg_names.append(p.name)

        (
            bound_args,
            sig_and_spec,
            constexpr_vals,
            non_constexpr_vals,
            excess_kwargs,
        ) = self.fn.binder(*args, **kwargs)
        bind_end = time.time()

        if callable(kwargs["grid"]):
            grid = kwargs["grid"](kwargs)
        else:
            grid = kwargs["grid"]

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)

        prepared_kernel = PreparedKernel(
            grid,
            kernel,
            launch_metadata,
            self.fn.CompiledKernel.launch_enter_hook,
            self.fn.CompiledKernel.launch_exit_hook,
            non_const_arg_names,
            self.cache_index_func(kwargs),
            device,
            stream,
        )

        wrapper_end = time.time()
        compile_time = compile_end - compile_start
        bind_time = bind_end - compile_end
        wrapper_time = wrapper_end - bind_end

        if flag_print_debug:
            print(
                f"[triton-dejavu] JIT compilation took {compile_time}s, binding {bind_time}, wrapper {wrapper_time}s."
            )

        return prepared_kernel

    def run(self, *args, **kwargs):
        # we only support kwargs
        assert len(args) == 0
        # assert no config pre-hook
        assert "pre_hook" not in kwargs or kwargs["pre_hook"] is None

        # print(f"my lock: {self.cache_lock.is_locked}")
        # TODO ?: or len(self.kernel_cache) == 0:
        if not self.cache_lock.is_locked:
            # we only support int, bool, float as cache index
            for key in self.check_keys:
                assert type(kwargs[key]) in [int, bool, float]
            prepared_kernel = self._get_prepared_kernel(*args, **kwargs)
            self.kernel_cache[prepared_kernel.get_key()] = prepared_kernel

        # TODO: if the cache index is not present, it will create an exception
        #  should we instead then compile it?
        kernel_variant = self.kernel_cache[self.cache_index_func(kwargs)]

        return kernel_variant(*args, **kwargs)


def jitcache(
    cache_lock,
    check_keys,
):
    """
    Decorator for caching a :code:`triton.jit`'d function.

    :param cache_lock: The CacheLock used for this JitCache.
    :type cache_lock: CacheLock
    :param check_keys: The list of tl.constexpr that are used to index the cache. Only types int, bool, float are supported.
    :type check_keys: list[str]
    """

    def decorator(fn):
        return JitCache(
            fn,
            fn.arg_names,
            cache_lock,
            check_keys,
        )

    return decorator
