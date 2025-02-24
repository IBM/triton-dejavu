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
import numpy as np
import torch

# import torch.multiprocessing as mp
import multiprocessing as mp
import io
from collections import namedtuple
import triton
from triton.runtime.driver import driver
import gc
import traceback

from .dejavu_utilities import (
    create_dir_if_not_exist_recursive,
    get_tmp_storage_path,
    flag_print_debug,
    flag_print_debug_verbose,
)

__separate_process_dump_file__ = (
    f"{get_tmp_storage_path()}/isolated_bench/dejavu-mp-dump.log"
)


class SerializeableCompiledKernel(triton.compiler.CompiledKernel):
    def __init__(self, compiled_kernel: triton.compiler.CompiledKernel):
        self.metadata = compiled_kernel.metadata
        self.name = compiled_kernel.name
        self.packed_metadata = compiled_kernel.packed_metadata
        # self.src = compiled_kernel.src
        # src is ASTSource, which can't be pickled entirely...but we only need some parts
        # src.fn.constexprs, constants, src.fn.arg_names, signature
        # TODO: check for other platforms?
        # TODO: check for CompiledKernel.launch_metadata
        ASTSourceLight = namedtuple(
            "ASTSource", sorted(["constants", "signature", "fn"])
        )
        JITFunctionLight = namedtuple(
            "JITFunction", sorted(["constexprs", "arg_names"])
        )
        fn_light = JITFunctionLight(
            constexprs=compiled_kernel.src.fn.constexprs,
            arg_names=compiled_kernel.src.fn.arg_names,
        )
        ast_src = ASTSourceLight(
            fn=fn_light,
            constants=compiled_kernel.src.constants,
            signature=compiled_kernel.src.signature,
        )
        self.src = ast_src
        # self._ast_src_dict = {'constants': self.src.constants, 'signature': self.src.signature,
        #                       'fn': {'constexprs': self.src.fn.constexprs, 'arg_names': self.src.fn.arg_names}}
        self.hash = compiled_kernel.hash
        self.asm = compiled_kernel.asm
        self.kernel = compiled_kernel.kernel
        self.module = None
        self.function = None

    def __getstate__(self):
        ast_src_dict = {
            "constants": self.src.constants,
            "signature": self.src.signature,
            "fn": {
                "constexprs": self.src.fn.constexprs,
                "arg_names": self.src.fn.arg_names,
            },
        }
        state = (
            self.metadata._asdict(),
            self.name,
            self.packed_metadata,
            ast_src_dict,
            self.hash,
            self.asm,
            self.kernel,
            self.module,
            self.function,
        )
        return state

    def __setstate__(self, state):
        (
            metadata_dict,
            self.name,
            self.packed_metadata,
            ast_src_dict,
            self.hash,
            self.asm,
            self.kernel,
            self.module,
            self.function,
        ) = state
        KernelMetadata = namedtuple(
            "KernelMetadata", sorted(list(metadata_dict.keys()))
        )
        ASTSourceLight = namedtuple("ASTSource", sorted(list(ast_src_dict.keys())))
        JITFunctionLight = namedtuple(
            "JITFunction", sorted(list(ast_src_dict["fn"].keys()))
        )
        self.metadata = KernelMetadata(**metadata_dict)
        fn_light = JITFunctionLight(**ast_src_dict["fn"])
        ast_src_dict["fn"] = fn_light
        ast_src = ASTSourceLight(**ast_src_dict)
        self.src = ast_src


class CompiledKernelRun:
    def __init__(
        self,
        grid_0,
        grid_1,
        grid_2,
        kernel,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *non_constexpr_vals,
    ):
        self.grid_0 = grid_0
        self.grid_1 = grid_1
        self.grid_2 = grid_2
        # streams etc. can't be pickled
        self.kernel = SerializeableCompiledKernel(kernel)
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.non_constsexpr_vals = non_constexpr_vals

    def __call__(self):
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        return self.kernel.run(
            self.grid_0,
            self.grid_1,
            self.grid_2,
            stream,
            self.kernel.function,
            self.kernel.packed_metadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *self.non_constsexpr_vals,
        )

    def get_stream(self):
        new_stream = torch.cuda.Stream()
        self.stream = new_stream
        return new_stream


class KernelEvalCall:
    def __init__(
        self,
        fn,
        arg_names,
        benchmarking_stream,
        cur_config,
        call_lambda,
        *args,
        **current,
    ):
        self.fn = fn
        # self.args = args
        # Necessary to avoid persistent RuntimeErrors
        self.args = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
        self.current = current
        self.arg_names = arg_names
        # run_args = [v for k,v in kernel_call.current.items() if k in arg_names]
        self.benchmarking_stream = benchmarking_stream
        self.call_lambda = call_lambda
        self.compiled_kernel = None
        self.cur_config = cur_config
        self._jit_was_triggered = False

    def __call__(self):
        # pre_hook, post_hook must be part of call_lambda (see autotuner.py)
        if not self._jit_was_triggered:
            self._jit_was_triggered = True

            def jit_first_time():
                compile_start = time.time()
                ret = self.call_lambda()
                compile_end = time.time()
                compile_time = compile_end - compile_start
                if flag_print_debug:
                    print(
                        f"[triton-dejavu] First execution including JIT compilation took {compile_time}s."
                    )
                return ret

            return jit_first_time()
        else:
            return self.call_lambda()

    def _call_config_pre_hook(self):
        # must be done before binding...so not separated...
        pre_hook = self.cur_config.pre_hook if self.cur_config.pre_hook else None
        if not pre_hook:
            return
        print(f"[triton-dejavu] Executing pre_hook of config {self.cur_config}...")
        prehook_start = time.time()
        nargs = dict(zip(self.arg_names, self.args))
        full_nargs = {**nargs, **self.current}
        pre_hook(full_nargs)
        prehook_end = time.time()
        prehook_duration = prehook_end - prehook_start
        print(f"\t...pre_hook done ({prehook_duration}s).")

    def get_stream(self):
        return self.benchmarking_stream

    def get_compiled_run(self) -> CompiledKernelRun:
        # need to call config pre-hook first...
        self._call_config_pre_hook()

        self.current["warmup"] = True
        compile_start = time.time()
        kernel = self.fn.run(*self.args, **self.current)
        compile_end = time.time()
        (
            bound_args,
            sig_and_spec,
            constexpr_vals,
            non_constexpr_vals,
            excess_kwargs,
        ) = self.fn.binder(*self.args, **self.current)
        bind_end = time.time()
        self._jit_was_triggered = True

        # device = triton.runtime.driver.active.get_current_device()
        # stream = triton.runtime.driver.active.get_current_stream(device)

        if callable(self.current["grid"]):
            grid = self.current["grid"](self.current)
        else:
            grid = self.current["grid"]
        launch_metadata = kernel.launch_metadata(
            grid, self.benchmarking_stream, *non_constexpr_vals
        )

        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1

        # kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
        #            self.fn.CompiledKernel.launch_enter_hook, self.fn.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        self.compiled_kernel = CompiledKernelRun(
            grid_0,
            grid_1,
            grid_2,
            kernel,
            launch_metadata,
            self.fn.CompiledKernel.launch_enter_hook,
            self.fn.CompiledKernel.launch_exit_hook,
            *non_constexpr_vals,
        )
        wrapper_end = time.time()
        compile_time = compile_end - compile_start
        bind_time = bind_end - compile_end
        wrapper_time = wrapper_end - bind_end

        if flag_print_debug:
            print(
                f"[triton-dejavu] JIT compilation took {compile_time}s, binding {bind_time}, wrapper {wrapper_time}s."
            )

        return self.compiled_kernel

    def cleanup(self):
        if hasattr(self, "args"):
            for a in self.args:
                if isinstance(a, torch.Tensor):
                    del a
            del self.args
        if hasattr(self, "compiled_kernel"):
            del self.compiled_kernel
        gc.collect()


def _do_bench_cudagraph(
    fn: KernelEvalCall,
    return_dict,
    rep=20,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
    redirect_io=False,
):
    """
    Similar to upstream, but we use an internal fork that has the same interface in the case of using cuda graphs or not.
    Also, it supports being lunched in a separate process
    """
    if redirect_io:
        # redirect below python level
        if flag_print_debug:
            os.dup2(os.open(__separate_process_dump_file__, os.O_APPEND), 1)
            os.dup2(os.open(__separate_process_dump_file__, os.O_APPEND), 2)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
    try:
        # print("starting _do_bench_cudagraph...\n")
        assert return_mode in ["min", "max", "mean", "median"]

        # TODO: use device, stream, and device interface once API settled in Triton
        #  (and we don't need trtion < 3.2 any more)
        # device = driver.active.get_current_device()
        # stream = driver.active.get_current_stream(device)
        # di = driver.active.get_device_interface()

        with torch.cuda.stream(fn.get_stream()):
            if torch.cuda.current_stream() == torch.cuda.default_stream():
                raise RuntimeError(
                    "Cannot capture graph in default stream. Please use side stream in benchmark code."
                )

            # warmup & JIT if not separated process
            fn()

            # We maintain a buffer of 256 MB that we clear
            # before each kernel call to make sure that the L2
            # doesn't contain any input data before the run
            # TODO: use driver.active... in the future
            if fast_flush:
                cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
            else:
                cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

            # step 1 - we estimate the amount of time the kernel call takes
            # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
            #       but it is probably good enough
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.detach_()
                    x.requires_grad_(True)
                    x.grad = None
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            estimate_ms = start_event.elapsed_time(end_event)
            n_repeat = max(1, int(rep / estimate_ms))
            # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
            #  host overhead
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n_repeat):
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    fn()
            torch.cuda.synchronize()
            # measure time and return
            ret = []
            n_retries = 10
            # NOTE: in contrast do upstream, we do clear L2 Cache
            #  However, also we can't do it within a cuda graph, where the kernel is launched n_repeat times
            for _ in range(n_retries):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                g.replay()
                end_event.record()
                torch.cuda.synchronize()
                ret += [start_event.elapsed_time(end_event) / n_repeat]
                cache.zero_()
            times = torch.tensor(ret)
            # return getattr(torch, return_mode)(times).item()
            if quantiles is not None:
                ret = torch.quantile(
                    times, torch.tensor(quantiles, dtype=torch.float)
                ).tolist()
                if len(ret) == 1:
                    ret = ret[0]
            else:
                ret = getattr(torch, return_mode)(times).item()
            return_dict["ret"] = ret
    except Exception as e:
        print(f"bench_cudagraph failed with {e}")
        # return_dict["e"] = e
        tb = traceback.format_exc()
        return_dict["e"] = f"Exception {e}; traceback: {tb}"
        print(tb)
    fn.cleanup()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if redirect_io:
        return_dict["stdout"] = sys.stdout.getvalue()
        return_dict["stderr"] = sys.stdout.getvalue()


def _do_bench_cuda_eager(
    fn: KernelEvalCall,
    return_dict,
    warmup=25,
    rep=100,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
    redirect_io=False,
):
    """
    Similar to upstream, but we use an internal fork that has the same interface in the case of using cuda graphs or not.
    Also, it supports being lunched in a separate process
    """
    assert return_mode in ["min", "max", "mean", "median"]
    import torch

    if redirect_io:
        # redirect below python level
        if flag_print_debug:
            os.dup2(os.open(__separate_process_dump_file__, os.O_APPEND), 1)
            os.dup2(os.open(__separate_process_dump_file__, os.O_APPEND), 2)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
    try:

        # TODO: use device, stream, and device interface once API settled in Triton
        #  (and we don't need trtion < 3.2 any more)
        # device = driver.active.get_current_device()
        # stream = driver.active.get_current_stream(device)
        # di = driver.active.get_device_interface()

        fn()
        torch.cuda.synchronize()

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2
        # doesn't contain any input data before the run
        # TODO: use driver.active... in the future
        if fast_flush:
            cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
        else:
            cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

        # Estimate the runtime of the function
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            cache.zero_()
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
        start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        # Warm-up
        for _ in range(n_warmup):
            fn()
        # Benchmark
        for i in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            start_event[i].record()
            fn()
            end_event[i].record()
        # Record clocks
        torch.cuda.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float,
        )
        if quantiles is not None:
            ret = torch.quantile(
                times, torch.tensor(quantiles, dtype=torch.float)
            ).tolist()
            if len(ret) == 1:
                ret = ret[0]
        else:
            ret = getattr(torch, return_mode)(times).item()
        return_dict["ret"] = ret
    except Exception as e:
        print(f"bench_cuda_eager failed with {e}")
        # return_dict["e"] = e
        tb = traceback.format_exc()
        return_dict["e"] = f"Exception {e}; traceback: {tb}"
        print(tb)
    fn.cleanup()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if redirect_io:
        return_dict["stdout"] = sys.stdout.getvalue()
        return_dict["stderr"] = sys.stdout.getvalue()


def do_bench(
    fn: KernelEvalCall,
    use_cuda_graphs=True,
    warmup=25,
    rep=20,
    grad_to_none=None,
    quantiles=None,
    return_mode="mean",
    use_isolated_process=False,
    run_id=0,
    path_prefix="tensor_dump",
    verify_out_index=None,
):
    """
    Benchmark the runtime of the provided function.

    :param fn: Function to benchmark
    :type fn: Callable
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", or "median". Default is "mean".
    :type return_mode: str
    """
    if not use_isolated_process:
        # (more or less) equivalent to trtion upstream
        return_dict = {"ret": float("nan")}
        if use_cuda_graphs:
            _do_bench_cudagraph(
                fn, return_dict, rep, grad_to_none, quantiles, False, return_mode
            )
        else:
            _do_bench_cuda_eager(
                fn,
                return_dict,
                warmup,
                rep,
                grad_to_none,
                quantiles,
                False,
                return_mode,
            )
        fn.cleanup()
        return return_dict["ret"]
    else:
        free_m, total_m = torch.cuda.mem_get_info()
        GB_u = 1024 * 1024 * 1024
        print(
            f"current memory: {free_m/GB_u:.4f} GB free of total {total_m/GB_u:.4f} GB. "
        )
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        return_dict = manager.dict({"ret": float("nan"), "stdout": "", "stderr": ""})
        compiled_fn = fn.get_compiled_run()
        if flag_print_debug:
            dir_name = os.path.dirname(__separate_process_dump_file__)
            create_dir_if_not_exist_recursive(dir_name)
            if not os.path.isfile(__separate_process_dump_file__):
                open(__separate_process_dump_file__, "a").close()
            try:
                os.chmod(__separate_process_dump_file__, 0o0777)
            except PermissionError as e:
                print(
                    f"can't set permission of file {__separate_process_dump_file__}: {e}"
                )
        if use_cuda_graphs:
            p = mp.Process(
                target=_do_bench_cudagraph,
                args=(
                    compiled_fn,
                    return_dict,
                    rep,
                    grad_to_none,
                    quantiles,
                    False,
                    return_mode,
                    True,
                ),
            )
        else:
            p = mp.Process(
                target=_do_bench_cuda_eager,
                args=(
                    compiled_fn,
                    return_dict,
                    warmup,
                    rep,
                    grad_to_none,
                    quantiles,
                    False,
                    return_mode,
                    True,
                ),
            )
        p.start()
        p.join()
        ret = return_dict["ret"]
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] separated process returned with {ret} [run {run_id:06d}] (stdout: {return_dict['stdout']})"
        )
        if (np.isnan(ret) or "e" in return_dict) and flag_print_debug:
            e = return_dict.get("e", "(unknown)")
            print(
                f"[triton-dejavu] [{time.strftime('%Y-%m-%d %H:%M:%S')}] benchmark process failed with: {e}; {return_dict['stderr']}"
            )
            # raise Exception(str(return_dict['e']))
            print("trying to kill the process...")
            # kill is necessary after `Triton Error [CUDA]: an illegal memory access was encountered;` ??
            #   doesn't work...
            p.kill()
            free_m, total_m = torch.cuda.mem_get_info()
            GB_u = 1024 * 1024 * 1024
            print(
                f"after kill: {free_m/GB_u:.4f} GB free of total {total_m/GB_u:.4f} GB. "
            )
        if not np.isnan(ret) and verify_out_index is not None:
            tensor_path = f"{get_tmp_storage_path()}/{path_prefix}/v0_{fn.fn.hash}-run{run_id:06d}-idx{verify_out_index}.npy"
            target_tensor = (
                compiled_fn.non_constsexpr_vals[verify_out_index].cpu().numpy()
            )
            if os.path.exists(tensor_path):
                # then we compare our result to existing:
                compare_tensor = np.load(tensor_path)
                # if not np.allclose(compare_tensor, target_tensor):
                #     print("result of separate process differ...")
                # ATOL = 1e-2
                ATOL = 0.015  # TODO: suffficient? how to generalize?
                triton.testing.assert_close(
                    compare_tensor, target_tensor, atol=ATOL, rtol=0
                )
            else:
                dir_name = os.path.dirname(tensor_path)
                create_dir_if_not_exist_recursive(dir_name)
                np.save(tensor_path, target_tensor)
                try:
                    os.chmod(tensor_path, 0o0777)
                except PermissionError as e:
                    print(f"can't set permission of file {tensor_path}: {e}")
        # this is required to free all associated GPU memory if `fn.cleanup()` couldn't/wasn't executed at the GPU side
        # (close is not enough)
        p.terminate()
        p.close()
        manager.shutdown()
        # to cleanup copy on this side
        fn.cleanup()
        del fn
        gc.collect()
        return ret
