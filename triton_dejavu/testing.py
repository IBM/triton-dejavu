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
import numpy as np
import torch
# import torch.multiprocessing as mp
import multiprocessing as mp
import io
from collections import namedtuple
import triton
from triton.runtime.driver import driver
import gc


#__separate_process_dump_file__ = '/tmp/dejavu-mp-dump.log'
__separate_process_dump_file__ = '/storage/tmp/dejavu-mp-dump.log'


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
        ASTSourceLight = namedtuple('ASTSource', sorted(['constants', 'signature', 'fn']))
        JITFunctionLight = namedtuple('JITFunction', sorted(['constexprs', 'arg_names']))
        fn_light = JITFunctionLight(constexprs=compiled_kernel.src.fn.constexprs, 
                                    arg_names=compiled_kernel.src.fn.arg_names)
        ast_src = ASTSourceLight(fn=fn_light, constants=compiled_kernel.src.constants, signature=compiled_kernel.src.signature)
        self.src = ast_src
        # self._ast_src_dict = {'constants': self.src.constants, 'signature': self.src.signature,
        #                       'fn': {'constexprs': self.src.fn.constexprs, 'arg_names': self.src.fn.arg_names}}
        self.hash = compiled_kernel.hash
        self.asm = compiled_kernel.asm
        self.kernel = compiled_kernel.kernel
        self.module = None
        self.function = None

    def __getstate__(self):
        ast_src_dict = {'constants': self.src.constants, 'signature': self.src.signature,
                              'fn': {'constexprs': self.src.fn.constexprs, 'arg_names': self.src.fn.arg_names}}
        state = (self.metadata._asdict(), self.name, self.packed_metadata, ast_src_dict, self.hash,
                 self.asm, self.kernel, self.module, self.function)
        return state
    
    def __setstate__(self, state): 
        metadata_dict, self.name, self.packed_metadata, ast_src_dict, self.hash, self.asm, \
            self.kernel, self.module, self.function = state
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata_dict.keys())))
        ASTSourceLight = namedtuple('ASTSource', sorted(list(ast_src_dict.keys())))
        JITFunctionLight = namedtuple('JITFunction', sorted(list(ast_src_dict['fn'].keys())))
        self.metadata = KernelMetadata(**metadata_dict)
        fn_light = JITFunctionLight(**ast_src_dict['fn'])
        ast_src_dict['fn'] = fn_light
        ast_src = ASTSourceLight(**ast_src_dict)
        self.src = ast_src
        


class CompiledKernelRun:

    def __init__(self, grid_0, grid_1, grid_2, stream, kernel, launch_metadata,
                   launch_enter_hook, launch_exit_hook, *non_constexpr_vals):
        self.grid_0 = grid_0
        self.grid_1 = grid_1
        self.grid_2 = grid_2
        # self.stream = stream
        # self.stream = torch.cuda.Stream()
        self.stream = None  # can't be pickled?
        self.kernel = SerializeableCompiledKernel(kernel)
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.non_constsexpr_vals = non_constexpr_vals

    def __call__(self):
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        return self.kernel.run(self.grid_0, self.grid_1, self.grid_2, stream, self.kernel.function, self.kernel.packed_metadata, 
                        self.launch_metadata, self.launch_enter_hook, self.launch_exit_hook, *self.non_constsexpr_vals)

    def get_stream(self):
        new_stream = torch.cuda.Stream()
        self.stream = new_stream
        return new_stream



class KernelEvalCall:

    def __init__(self, fn, arg_names, benchmarking_stream, call_lambda, *args, **current):
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

    def __call__(self):
        # TODO: config pre hook, post hook
        # return self.fn.run(*self.args, **self.current)
        return self.call_lambda()

    def get_stream(self):
        return self.benchmarking_stream

    def get_compiled_run(self) -> CompiledKernelRun:
        # kernel = self.fn.run(*self.args, warmup=True, **self.current)
        self.current['warmup'] = True
        kernel = self.fn.run(*self.args, **self.current)
        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.fn.binder(*self.args, **self.current)

        # device = triton.runtime.driver.active.get_current_device()
        # stream = triton.runtime.driver.active.get_current_stream(device)

        grid = self.current['grid'](self.current)
        launch_metadata = kernel.launch_metadata(grid, self.benchmarking_stream, *non_constexpr_vals)

        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1

        # kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
        #            self.fn.CompiledKernel.launch_enter_hook, self.fn.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        self.compiled_kernel = CompiledKernelRun(grid_0, grid_1, grid_2, self.benchmarking_stream, kernel, launch_metadata,
                                            self.fn.CompiledKernel.launch_enter_hook, self.fn.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        
        return self.compiled_kernel

    def cleanup(self):
        if hasattr(self, 'args'):
            for a in self.args:
                if isinstance(a, torch.Tensor):
                    del a
            del self.args
        if hasattr(self, 'compiled_kernel'):
            del self.compiled_kernel
        gc.collect()


def _do_bench_cudagraph(fn, return_dict, rep=20, grad_to_none=None, return_mode="mean", redirect_io=False):
    if redirect_io:
        # redirect below python level
        if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
            # os.dup2(os.open(os.devnull, os.O_RDWR), 1)
            # os.dup2(os.open(os.devnull, os.O_RDWR), 2)
            os.dup2(os.open(__separate_process_dump_file__, os.O_APPEND), 1)
            os.dup2(os.open(__separate_process_dump_file__, os.O_APPEND), 2)
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
    try:
        # print("starting _do_bench_cudagraph...\n")
        assert return_mode in ["min", "max", "mean", "median"]

        with torch.cuda.stream(fn.get_stream()):
            if torch.cuda.current_stream() == torch.cuda.default_stream():
                raise RuntimeError("Cannot capture graph in default stream. Please use side stream in benchmark code.")
            # warmup
            fn()
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
            # host overhead
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
            for _ in range(n_retries):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                g.replay()
                end_event.record()
                torch.cuda.synchronize()
                ret += [start_event.elapsed_time(end_event) / n_repeat]
            times = torch.tensor(ret)
            # return getattr(torch, return_mode)(times).item()
            return_dict['ret'] = getattr(torch, return_mode)(times).item()
    except Exception as e:
        print(f'bench_cudagraph failed with {e}')
        return_dict['e'] = e
    fn.cleanup()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if redirect_io:
        return_dict['stdout'] = sys.stdout.getvalue()
        return_dict['stderr'] = sys.stdout.getvalue()


def test_f(return_dict):
    # import sys
    sys.stdout = io.StringIO()
    print('hello')
    return_dict['ret'] = 'hello'
    return_dict['stdout'] = sys.stdout



def do_bench_cudagraph(fn, rep=20, grad_to_none=None, return_mode="mean", use_isolated_process=False, run_id=0, path_prefix='tmp'):
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
        # FIXME
        # equivalent to trtion upstream
        return_dict = {'ret': float('nan')}
        _do_bench_cudagraph(fn, return_dict, rep, grad_to_none, return_mode)
        fn.cleanup()
        return return_dict['ret']
    else:
        free_m, total_m = torch.cuda.mem_get_info()
        GB_u = 1024 * 1024 * 1024
        print(f"current memory: {free_m/GB_u:.4f} GB free of total {total_m/GB_u:.4f} GB. ")
        # TODO make once? reduce overhead...
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        return_dict = manager.dict({'ret': float('nan'), 'stdout': '', 'stderr': ''})
        # from torch.multiprocessing.spawn import spawn
        # mp.spawn(_do_bench_cudagraph, args=(fn, return_dict, rep, grad_to_none, return_mode, True), nprocs=1, join=True, start_method='spawn')
        compiled_fn = fn.get_compiled_run()
        if os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
            dir_name = os.path.dirname(__separate_process_dump_file__)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, 0o0777)
            if not os.path.isfile(__separate_process_dump_file__):
                open(__separate_process_dump_file__, 'a').close()
        p = mp.Process(target=_do_bench_cudagraph, args=(compiled_fn, return_dict, rep, grad_to_none, return_mode, True))
        p.start()
        p.join()
        ret = return_dict['ret']
        print(f"separated process returned with {ret} [run {run_id:06d}] (stdout: {return_dict['stdout']})")
        # if len(return_dict['stderr']) > 0 and os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
        if (np.isnan(ret) or 'e' in return_dict) and os.environ.get("TRITON_DEJAVU_DEBUG", "0") == "1":
            e = return_dict.get('e', '(unknown)')
            print(f"[triton-dejavu] benchmark process failed with: {e}; {return_dict['stderr']}")
            # raise Exception(str(return_dict['e']))
            print("trying to kill the process...")
            # kill is necessary after `Triton Error [CUDA]: an illegal memory access was encountered;` ??
            #   doesn't work...
            p.kill()
            free_m, total_m = torch.cuda.mem_get_info()
            GB_u = 1024 * 1024 * 1024
            print(f"after kill: {free_m/GB_u:.4f} GB free of total {total_m/GB_u:.4f} GB. ")
        if not np.isnan(ret):
            tensor_path = f"/storage/tensor_dump/{path_prefix}/v0_{fn.fn.hash}-run{run_id:06d}.npy"
            target_tensor = compiled_fn.non_constsexpr_vals[2].cpu().numpy()
            if os.path.exists(tensor_path):
                # then we compare our result to existing:
                compare_tensor = np.load(tensor_path)
                # if not np.allclose(compare_tensor, target_tensor):
                #     print("result of separate process differ...")
                # ATOL = 1e-2
                ATOL = 0.015  # TODO: suffficient? how to generalize?
                triton.testing.assert_close(compare_tensor, target_tensor, atol=ATOL, rtol=0)
            else:
                dir_name = os.path.dirname(tensor_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, 0o0777)
                np.save(tensor_path, target_tensor)
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


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", or "median". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median"]

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(cache_size), dtype=torch.int8, device='cuda')

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
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()
