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

import sys
import os

import pytest
import torch
import torch.nn as nn
import triton
import pandas as pd
from typing import Tuple
import triton_dejavu


# CUDA_DEVICES = [
#     f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
# ]
CUDA_DEVICES = ["cuda:0"]

DTYPES = [torch.float16]
SEEDS = [0]

NUM_TOKENS = [7, 144, 2048]
HIDDEN_SIZES = [768, 4096, 8192]

BATCH_SIZE_FLASH = [1, 32]
NUM_HEADS_FLASH = [2, 32]
SEQUENCE_LENGTH_FLASH = [512, 2048]
HEAD_SIZES_FLASH = [32, 64, 128]  # only powers of 2!
CAUSAL = [True]  # vLLM only needs causal=True
VARLEN = [True]  # vLLM only needs varlen=True
MAX_VALUE = [0.01, 1.0, 100.0]
MAX_VALUE_FLASH = [0.01, 1.0]


do_benchmarks = True
quantiles = [0.5, 0.2, 0.8]
dump_dejavu_storage = True


# based on https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_layernorm.py (Apache 2.0 license)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("max_value", MAX_VALUE)
@torch.inference_mode()
def test_rms_norm(
    capsys,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_value: float,
) -> None:
    from rms_norm import rmsnorm_triton_wrapper

    ATOL = 1e-4
    RTOL = 1e-3 * 2

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    tdev = torch.device(device)
    # print(tdev)
    layer_weights = torch.ones(hidden_size)
    layer_weights.normal_(mean=1.0, std=0.1)
    layer_weights = layer_weights.to(device=tdev, dtype=dtype)
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=tdev).uniform_(
        -1 * max_value, max_value
    )
    x *= scale

    # PyTorch-native implementation equivalent to forward()
    orig_dtype = x.dtype
    xt = x.to(dtype=torch.float32)
    variance = xt.pow(2).mean(dim=-1, keepdim=True)
    xt = xt * torch.rsqrt(variance + 1e-6)
    xt = xt.to(dtype=orig_dtype, device=tdev)
    xt = xt * layer_weights
    ref_out = xt

    # test triton kernel
    out = rmsnorm_triton_wrapper(x, layer_weights)

    triton.testing.assert_close(ref_out, out, atol=ATOL, rtol=RTOL)

    captured = ""
    if capsys is not None:
        captured_raw = capsys.readouterr()
        for l in captured_raw:
            if len(l) > 0:
                captured += l + " "

    # benchmark only correct results
    if do_benchmarks:
        my_name = sys._getframe().f_code.co_name
        if my_name not in pytest.global_pds:
            pytest.global_pds[my_name] = pd.DataFrame(
                columns=[
                    "num_tokens",
                    "hidden_size",
                    "dtype",
                    "device",
                    "max_value",
                    "ms",
                    "min_ms",
                    "max_ms",
                    "captured",
                ]
            )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rmsnorm_triton_wrapper(x, layer_weights), quantiles=quantiles
        )
        nr = [
            num_tokens,
            hidden_size,
            dtype,
            device,
            max_value,
            ms,
            min_ms,
            max_ms,
            captured,
        ]
        pytest.global_pds[my_name].loc[len(pytest.global_pds[my_name])] = nr

    # cleanup memory
    del x
    del out
    del ref_out
    torch.cuda.empty_cache()


# based on https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py (Apache 2.0 license)
@pytest.mark.parametrize("batch_size", BATCH_SIZE_FLASH)
@pytest.mark.parametrize("num_heads", NUM_HEADS_FLASH)
@pytest.mark.parametrize("seqlen", SEQUENCE_LENGTH_FLASH)
@pytest.mark.parametrize("head_size", HEAD_SIZES_FLASH)
@pytest.mark.parametrize("causal", CAUSAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("varlen_mode", VARLEN)
@pytest.mark.parametrize("max_value", MAX_VALUE_FLASH)
@torch.inference_mode()
def test_flash_attention_v2(
    capsys,
    batch_size,
    num_heads,
    seqlen,
    head_size,
    causal,
    dtype,
    seed,
    device,
    varlen_mode,
    max_value,
):
    from rocm_flash_attention import attn_forward_wrapper

    ATOL = 1e-2 * max_value
    RTOL = 1e-2
    torch.manual_seed(seed)
    tdev = torch.device(device)
    torch.cuda.set_device(tdev)

    q = None
    k = None
    v = None
    M = None
    p = None
    prompt_lens = None
    prompt_lens_tensor = None
    out = None
    ref_out = None
    cu_seqlens_q = None
    seq_start_loc = None
    cu_seqlens_k = None

    inner_exception = None
    try:
        # q = (torch.empty((batch_size, num_heads, seqlen, head_size), dtype=dtype, device=tdev).normal_(mean=0.0, std=0.5).requires_grad_())
        # k = (torch.empty((batch_size, num_heads, seqlen, head_size), dtype=dtype, device=tdev).normal_(mean=0.0, std=0.5).requires_grad_())
        # v = (torch.empty((batch_size, num_heads, seqlen, head_size), dtype=dtype, device=tdev).normal_(mean=0.0, std=0.5).requires_grad_())
        q = (
            torch.empty(
                (batch_size, num_heads, seqlen, head_size), dtype=dtype, device=tdev
            )
            .normal_(mean=0.0, std=0.5 * max_value)
            .requires_grad_()
        )
        k = (
            torch.empty(
                (batch_size, num_heads, seqlen, head_size), dtype=dtype, device=tdev
            )
            .normal_(mean=0.0, std=0.5 * max_value)
            .requires_grad_()
        )
        v = (
            torch.empty(
                (batch_size, num_heads, seqlen, head_size), dtype=dtype, device=tdev
            )
            .normal_(mean=0.0, std=0.5 * max_value)
            .requires_grad_()
        )
        sm_scale = 0.5
        # dout = torch.randn_like(q)

        # reference implementation
        M = torch.tril(torch.ones((seqlen, seqlen), device=tdev))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        # p = torch.exp(p)
        ref_out = torch.matmul(p, v)

        # triton implementation
        cu_seqlens_q = None
        cu_seqlens_k = None
        if varlen_mode:
            # with varlen
            # q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
            # k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            # v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            # cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            #    of the sequences in the batch, used to index into q.
            # cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            #    of the sequences in the batch, used to index into kv.
            # max_seqlen_q: int. Maximum query sequence length in the batch.
            # max_seqlen_k: int. Maximum key sequence length in the batch.
            # out: (total, nheads, headdim).
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            q = q.reshape(batch_size * seqlen, num_heads, head_size)
            k = k.reshape(batch_size * seqlen, num_heads, head_size)
            v = v.reshape(batch_size * seqlen, num_heads, head_size)
            # # based on https://github.com/ROCm/triton/blob/b9e5290de8bf3a79c4e91ceed7e61b3c8d041b30/python/perf-kernels/flash-attention.py#L1149
            # seqlens_q = torch.randint(1, seqlen + 1, (batch_size,), dtype=torch.int32)
            # seqlens_k = torch.randint(1, seqlen + 1, (batch_size,), dtype=torch.int32)
            # cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(dim=0, dtype=torch.int32)])
            # cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_k.cumsum(dim=0, dtype=torch.int32)])
            # cu_seqlens_q = cu_seqlens_q.to(tdev)
            # cu_seqlens_k = cu_seqlens_k.to(tdev)
            # from above...
            seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=tdev)
            prompt_lens = [seqlen] * batch_size
            prompt_lens_tensor = torch.tensor(
                prompt_lens, dtype=torch.long, device=tdev
            )
            torch.cumsum(
                prompt_lens_tensor,
                dim=0,
                dtype=seq_start_loc.dtype,
                out=seq_start_loc[1:],
            )
            cu_seqlens_q = seq_start_loc
            cu_seqlens_k = seq_start_loc
        out = attn_forward_wrapper(
            q, k, v, causal, sm_scale, seqlen, seqlen, cu_seqlens_q, cu_seqlens_k
        )
        if varlen_mode:
            out = out.unflatten(0, (batch_size, seqlen))
            out = out.transpose(1, 2)

        # for better reports
        triton.testing.assert_close(ref_out, out, atol=ATOL, rtol=RTOL)

        captured = ""
        if capsys is not None:
            captured_raw = capsys.readouterr()
            for l in captured_raw:
                if len(l) > 0:
                    captured += l + " "

        # benchmark only correct results
        if do_benchmarks:
            my_name = sys._getframe().f_code.co_name
            if my_name not in pytest.global_pds:
                pytest.global_pds[my_name] = pd.DataFrame(
                    columns=[
                        "batch_size",
                        "num_heads",
                        "seqlen",
                        "head_size",
                        "causal",
                        "varlen",
                        "dtype",
                        "device",
                        "max_value",
                        "ms",
                        "min_ms",
                        "max_ms",
                        "captured",
                    ]
                )
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: attn_forward_wrapper(
                    q,
                    k,
                    v,
                    causal,
                    sm_scale,
                    seqlen,
                    seqlen,
                    cu_seqlens_q,
                    cu_seqlens_k,
                ),
                quantiles=quantiles,
            )
            nr = [
                batch_size,
                num_heads,
                seqlen,
                head_size,
                causal,
                varlen_mode,
                dtype,
                device,
                max_value,
                ms,
                min_ms,
                max_ms,
                captured,
            ]
            pytest.global_pds[my_name].loc[len(pytest.global_pds[my_name])] = nr
    except Exception as e:
        print(e)
        inner_exception = e
    finally:
        # cleanup memory
        try:
            del q
            del k
            del v
            del p
            del M
            del out
            del ref_out
            del prompt_lens
            del prompt_lens_tensor
            del cu_seqlens_q
            del seq_start_loc
            del cu_seqlens_k
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(e)
        finally:
            if inner_exception is not None:
                raise inner_exception


if __name__ == "__main__":
    used_dejavu_storage = None
    if dump_dejavu_storage:
        from triton_dejavu import global_dejavu_storage

        used_dejavu_storage = global_dejavu_storage

    global_pds = {}
    gpu_name = torch.cuda.get_device_name()
    cuda_version = triton_dejavu.dejavu_utilities._get_cuda_version()
    print(
        f"\nRunning on {gpu_name} with Triton {triton.__version__} using cuda {cuda_version}...\n"
    )
    if do_benchmarks:
        pytest.do_benchmarks = do_benchmarks
        pytest.global_pds = global_pds
    if len(sys.argv) >= 1:
        args = [__file__]
        filter_args = ""
        for ca in sys.argv[1:]:
            if ca[0] == "-":
                args.append(ca)
            else:
                filter_args += f"{ca} or "
        if len(filter_args) > 2:
            args.append(f"-k {filter_args[:-3]}")
        pytest.main(args=args)
    else:
        pytest.main(args=[__file__])
    if do_benchmarks:
        for test, df in pytest.global_pds.items():
            print(
                f"\nPerformance results of test {test} (only tests without numerical error and with valid shapes, etc.):"
            )
            if os.environ.get("TRITON_DEJAVU_DEBUG_SAVE_CSV", "0") == "1":
                filename = "perf_test.csv"
                df.to_csv(filename, sep="\t", encoding="utf-8")
                print(f"(stored in {filename})")
            else:
                print(df.to_string())
        print(
            f"\nThis test used triton version: {triton.__version__}\n"
            f"This test was executed on: {gpu_name}\n"
            f"This test used cuda (nvcc): {cuda_version}"
        )
    if dump_dejavu_storage:
        print("\n\ndump dejavu storage:")
        used_dejavu_storage.dump_storage()
