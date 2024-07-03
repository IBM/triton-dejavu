Triton Deja-vu
=================
Framework to reduce the autotuner overhead of [triton-lang](https://github.com/triton-lang/triton) to (close to) 0 for well known deployments.

This small framework is based on the [Triton autotuner](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py) and contributes two features to the Triton community:
1. Store and safely restore autotuner states using JSON files. 
2. `ConfigSpaces` to explore a defined space exhaustively.

(more details see [below](#features))


Installation
----------------

Currently, triton-dejavu can only be installed from source:

```
git clone https://github.com/IBM/triton-dejavu.git
pip install -e triton-dejavu/
```


Usage
-----------------

To use the store and restore feature, simply replace the triton autotuner with the triton-dejavu autotuner:
```
import triton_dejavu

@triton_dejavu.autotune(
    ...
```
Second, the environment variable `TRITON_DEJAVU_STORAGE` needs to be set and point to a read and writable directory. 


To use the `ConfigSpaces` feature, replace the `config=` parameter for the triton_dejavu autotuner with  `config_space` definition:
```
 config_space=triton_dejavu.ConfigSpace(
        {'BLOCK_N_SIZE': [1024, 2048, 4096]},
        num_warps=[4, 8, 16],
        num_stages=[1, 2, 4, 6],
        num_ctas=[1],
        enable_warp_specialization=[False, True]
    ),
```

Examples
----------------

This repository contains two example Triton kernels in the `tests` directory using the provided Dockerfile:

```
docker build -f tests/Dockerfile . -t test-triton-dejavu

# create a directory to read & write the autotuner cache
chmod o+rw dejavu-data/

# run the container
docker run --rm -it --gpus '"device=0"' -v $(pwd)/dejavu-data/:/storage/dejavu-data/ test-triton-dejavu:latest
```


Features
----------------

### Store and safely restore autotuner states

Triton comes with a built-in autotuner, which is crucial to enable performance-portability. However, using the autotuner adds a lot more overhead to the kernel launches, in addition to the just-in-time compilation. This overhead comes from the fact that, for every variation in the kernel parameters, the autotuner needs to determine which kernel version performs the best. The resulting high variance in latency is unacceptable for serving applications in production. 
Additionally, we learned that for more complex kernels, the autotuner needs to choose from more options increasing the latency further.
Consequently, the Triton autotuner is usually not used in production today. Yet, by not using it, the portability of the application is limited, because the performance of the Triton kernels can differ by more than one order of magnitude on different platforms. 

To solve this problem, we have developed a “dejavu” mechanism for the Triton autotuner. Our goal was to let the autotuner “remember” earlier executions of the kernel, which happened before the lifetime of the current deployment. This dejavu-mechanism reduces the overhead of the Triton autotuner to zero and therefore enables the usage of the autotuner in production. 

Our triton-dejavu autotuner is based on the [upstream autotuner](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py) but additionally saves and restores already known cache states. In case the autotuner is triggered, and the restored cache does not contain the required key, a autotune run is executed, exactly as the original autotuner does.

To determine if a previously stored cache is still applicable, we use the combination of multiple values:

- cuda runtime version
- pytorch version
- triton version
- GPU type
- hash of the JIT function (i.e. `JITFunction.fn.hash`)
- hash of the autotuner key list
- hash of the configurations provided to the autotuner
- hash of some autotuner optional parameter

So far, we think that the above listed combination determines the applicability of a cache unambiguous. Hence, if all these values match a stored cache, this cache is restored and reused.

In addition, users can define a tag to be used by the dejavu storage to be able to differentiate different deployment scenarios (for otherwise identical value combinations).

Below is a simple example of how such a stored cache looks like (the “some_function” in the identifier is just there to help us humans analyzing what’s in the cache):

```
DejavuStorage identifier: dejavu_0.1/cuda_12.1/torch_2.1.2+cu121/triton_3.0.0/gpu_NVIDIA_A100_80GB_PCIe 
Cache identifier: some_function-ab4979d1539b394f48fe313d5462dc9254ae1623050232bd8d11077553c70c0c/default
Stored cache: 
{
        "signature": "JITFunction(some_function)",
        "total_bench_time_s": 23.483010053634644,
        "evaluated_configs": 16,
        "cache": {
            "(2, 2, True, 0.0, 0, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float32', 'torch.float16', 'torch.int32', 'torch.int32')": "BLOCK_M: 64, BLOCK_N: 64, num_warps: 8, num_ctas: 1, num_stages: 1",
            "(32, 32, True, 0.0, 0, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float32', 'torch.float16', 'torch.int32', 'torch.int32')": "BLOCK_M: 128, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 1"
        }
}
```

Our experiments show that our dejavu-autotuner removes the additional overhead of many autotuner configurations while still profiting from Tritons flexibility and increased performance.

### `ConfigSpaces`


The Triton autotuner requires that the developer provides a list of configuration options for different kernel variants to choose from. The selection of this list has a significant impact on the resulting performance. For example, our results show a difference of nearly 20x for complex kernels.
Hence, the dejavu-mechanism enabled us to develop a method for exploring the complete space of possible autotuner configurations leading to even better performance and reducing the amount of application-specific expert-knowledge required to effectively use Triton in production. 

The `ConfigSpaces` class allows to define ranges of parameters and then creates a list of all possible combinations. This list is then passed to the autotuner.
During generation of the list, configuration options that are only available on certain platforms are sorted out automatically. To facilitate this filtering, the user can specify a list of lambda functions using the optional `kwarg_conditions` parameter, as shown in the example below. Only configurations that fulfill all provided conditions are forwarded to the autotuner. 

```
 config_space=triton_dejavu.ConfigSpace(
        {
            'BLOCK_M': [32, 64, 128],
            'BLOCK_N': [32, 64, 128],
            'PRE_LOAD_V': [False]
        }, 
        kwarg_conditions = [lambda kwarg: kwarg['BLOCK_M'] >= kwarg['BLOCK_N'],
                           lambda kwarg: kwarg['BLOCK_M'] != 64 or gpu_name != 'NVIDIA H100 PCIe',
                           ],
        num_warps=[2, 4, 8],
        num_stages=[2, 4, 8],
        num_ctas=[1],  
        enable_warp_specialization=[False, True],  # for triton < 3.0
    ),
```

