Triton Deja-vu
=================
Framework to reduce autotune overhead of [triton-lang](https://github.com/triton-lang/triton) to zero for well known deployments.

This small framework is based on the [Triton autotuner](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py) and contributes three features to the Triton community:
1. Store and safely restore autotuner states using JSON files.
2. `ConfigSpaces` to explore a defined space exhaustively.
3. Bayesian Optimization to speed up the autotuning process.

Additionally, it allows to use heuristics in combination with the autotuner. Please find more details in the [feature section below](#features).

Besides improvements for autotuning, it also contains useful tools in working with triton, specifically a [cache for JIT-artifacts](#jitcache).


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


To use the `ConfigSpaces` feature, replace the `config` parameter for the triton_dejavu autotuner with  `config_space` definition:
```
 config_space=triton_dejavu.ConfigSpace(
        {'BLOCK_N_SIZE': [1024, 2048, 4096]},
        num_warps=[4, 8, 16],
        num_stages=[1, 2, 4, 6],
    ),
```

Examples
----------------

This repository contains two example Triton kernels in the `tests` directory using the provided Dockerfile:

```
docker build -f tests/Dockerfile . -t test-triton-dejavu

# create a directory to read & write the autotuner cache
mkdir dejavu-data/
chmod o+rw dejavu-data/

# run the container
docker run --rm -it --gpus '"device=0"' -v $(pwd)/dejavu-data/:/storage/dejavu-data/ test-triton-dejavu:latest
```

You can add e.g. `--build-arg triton_version=release/3.2.x` to the docker build command if you want to test not the latest `release` of triton.


Features
----------------

### Store and safely restore autotuner states

Triton comes with a built-in autotuner, which is crucial to enable performance-portability. However, using the autotuner adds a lot more overhead to the kernel launches, in addition to the just-in-time compilation. This overhead comes from the fact that, for every variation in the kernel parameters, the autotuner needs to determine which kernel version performs the best. The resulting high variance in latency is unacceptable for serving applications in production. 
Additionally, we learned that for more complex kernels, the autotuner needs to choose from more options increasing the latency further.
Consequently, the Triton autotuner is usually not used in production today. Yet, by not using it, the portability of the application is limited, because the performance of the Triton kernels can differ by more than one order of magnitude on different platforms. 

To solve this problem, we have developed a “dejavu” mechanism for the Triton autotuner. Our goal was to let the autotuner “remember” earlier executions of the kernel, which happened before the lifetime of the current deployment. This dejavu-mechanism reduces the overhead of the Triton autotuner to zero and therefore enables the usage of the autotuner in production. 

Our triton-dejavu autotuner is based on the [upstream autotuner](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py) but additionally saves and restores already known cache states. In case the autotuner is triggered, and the restored cache does not contain the required key, a autotune run is executed, exactly as the original autotuner does.

To determine if a previously stored cache is still applicable, we use the combination of multiple values:

- cuda runtime version (i.e. the ptxas used by triton) / rocm runtime version (i.e. the rocm ldd used by triton)
- triton version
- GPU type
- hash of the JIT function (i.e. `JITFunction.fn.hash`, but *without* the starting line number or global dependencies)
- hash of the autotuner key list
- hash of the configurations provided to the autotuner
- hash of some autotuner optional parameter
- (_major.minor version_ of triton-dejavu)

So far, we think that the above listed combination determines the applicability of a cache unambiguous. Hence, if all these values match a stored cache, this cache is restored and reused.

In addition, users can define a tag to be used by the dejavu storage to be able to differentiate different deployment scenarios (for otherwise identical value combinations).

Please note, the above list does not include features that do not influence the decision of the autotuner, but influence the behaviour of the kernel or the JIT. For example, the presence or details of `pre_hook` or `post_hook` and also other [`specialization_data`](https://github.com/triton-lang/triton/blob/e87f877eb94efeaeb4ad8697f315932121dec5e0/python/triton/runtime/jit.py#L514) used by the JIT cache are not used by triton-dejavu. 


#### Example

Below is a simple example of how such a stored cache looks like (the “some_function” in the identifier is just there to help us humans analyzing what’s in the cache):

```
DejavuStorage identifier: dejavu_0.7/triton_3.2.0/cuda_12.4/gpu_NVIDIA_A100_80GB_PCIe 
Cache identifier: some_function/autotune_config-9cefb332ef1d4228aeabeeb71300a10e49af618945049a462862f7ebcba76770/code_version-476f4fd1e55ef79ed14f270b5f9e7b6c5b4b0c0dbdc462ca3c98669dbb80a1b6/tune_features-df62f53ce178f143b59631de953c946e43811ff1b34cd71e422dfdf14ac35bb9/kernel_configs-55a194aa00a30b006356930f070398dd06fd7e3442e00591250f93f7fdb3e9de/default


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

#### Known Limitations

Although we think triton-dejavu is safe to use for most use cases, there is currently one caveat: 

Configuration pruning: If `prune_configs_by` is used, the triton kernel configurations passed to the autotuner are pruned for every autotuner run, which severely changes the decision of the autotuner. However, triton-dejavu can not capture the pruning function and its output in a way that would allow the safe re-store of it. Therefore, as of now, a user is responsible to ensure the use of the pruning function in combination with autotune cache restore is safe.


### `ConfigSpaces`


The Triton autotuner requires that the developer provides a list of configuration options for different kernel variants to choose from. The selection of this list has a significant impact on the resulting performance. For example, our results show a difference of nearly 20x for complex kernels.
Hence, the dejavu-mechanism enabled us to develop a method for exploring the complete space of possible autotuner configurations leading to even better performance and reducing the amount of application-specific expert-knowledge required to effectively use Triton in production. 

The `ConfigSpaces` class allows to define ranges of parameters and then creates a list of all possible combinations. This list is then passed to the autotuner.
During generation of the list, configuration options that are only available on certain platforms are sorted out automatically. To facilitate this filtering, the user can specify a list of functions using the optional `kwarg_conditions` parameter, as shown in the example below. The functions are then called with all instances of generated kwarg dictionaries. Only configurations where all functions evaluate to true are forwarded to the autotuner. 

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
    ),
```

### Configuration passthrough

The autotuner of triton-dejavu checks if the provided `kwargs` of a triton kernel invocation contains configuration parameters. If yes, the autotuner run is skipped and the provided configuration is used. This feature was added for situations where the application can provide configurations in some circumstances and therefore the autotuner has to be disabled in some cases but not all. 

### Fallback heuristic

To avoid additional latency of an autotuner run, triton-dejavu allows the restoring of previous autotuner caches. However, these caches may not contain all possible combinations of the autotuner keys. To avoid the random trigger of autotuner runs in latency-sensitive environments, users can provide heuristics to compute the configuration values with the flag `fallback_heuristic` (similarly to the [`@triton.heuristics` decorator](https://github.com/triton-lang/triton/blob/194a00f0f54ecd85dba202d840242c5f3f72b068/python/triton/runtime/autotuner.py#L340-L358)).
The provided heuristic is used if there is 1) no corresponding entry to the current `key`-tuple **and** 2) the environment variable `TRITON_DEJAVU_FORCE_FALLBACK=1` is set. 
The heuristic callable is then called with the current `key`-tuple as argument. 

```
@triton_dejavu.autotune(
    ...
    fallback_heuristic = lambda key: triton.Config({'BLOCK_SIZE': 2048 if key[1] <= 128 else 4096}, num_warps=16, num_stages=2),
    ...
```

If the environment variable `TRITON_PRINT_AUTOTUNING` is set, a log message about the use and outcome of the heuristic is printed. 


### Bayesian Optimization to speed up autotune process

Triton-dejavu can use [Bayesian Optimization (BO)](https://en.wikipedia.org/wiki/Bayesian_optimization) to speed up the tuning of kernels with very large search spaces. 
Triton-dejavu also implemented random search, since BO does not always convert quicker. 

Both features can be enabled with additional parameters to `triton_dejavu.autotune()`:
- `use_bo`: Activate Bayesian Optimization (BO) to speed up autotuner runs (at the expense of allowing some percentage of performance drop of the chosen kernel). This feature can only be used in combination with config_space. Also, prune_configs_by must not be provided.
- `use_random_search`: Activate Random Search to speed up autotuner runs (at the expense of allowing some percentage of performance drop of the chosen kernel). This feature can be used in combination with config_space and config lists. However, prune_configs_by must not be provided.
- `search_max_search_t`: Maximum search time (in seconds) for BO and Random Search.
- `search_max_share`: Maximum percentage of the total config space BO and Random Search can search through. This translates into a maximum trial number for the optimizer.

For the BO search, triton-dejavu depends on the [SMAC library](https://github.com/automl/SMAC3) and `numpy`, see `requirements-opt.txt`. These dependencies for BO sarch can also be installed via:
```
pip install "triton-dejavu[BO] @ file:./triton-dejavu"
```
Please note that smac depends on [swig](https://www.swig.org), which need to be installed first.


### JITCache

The launch overhead of triton kernels is a well known problem (see e.g. [1](https://github.com/triton-lang/triton/pull/3503), [2](https://github.com/triton-lang/triton/issues/2637), [3](https://github.com/triton-lang/triton/issues/6064)). Parts of the launch overhead comes from the fact that the triton JIT checks very carefully if an existing binary is safe to use.

In many scenarios, these checks can be relaxed. Such a cache with relaxed checks is implemented by `triton_dejavu.jitcache`. It is implemented as a decorator that could be used in front of the `triton.jit` decorator:

```
@triton_dejavu.jitcache(
    check_keys=["x", "BLOCK_SIZE", "USE_ALIBI_SLOPES", "SLIDING_WINDOW", "filter_by_query_len"],
)
@triton.jit
def kernel_paged_attention_...
```

The required `check_keys` argument must provide a list of the kernel parameters marked as `tl.constexpr` that **must be checked** to select the correct kernel binary. Ideally, this is just a subset of all constant kernel parameters.
For example, if we have two constant parameters A and B, but we know that A never will change in a particular application, but B will, then the list should look like `check_keys=["A"]`.

Consequently, *the usage of `triton_dejavu.jitcache` is application specific* (and also *experimental*).

Additionally, a user could provide a lock with e.g. `cache_lock=triton_dejavu.global_cache_lock` to ensure that no re-compilation happens after the cache lock is locked.

The `triton_dejavu.jitcache` reduces the launch overhead of triton kernels to 30-40 micro-seconds.


Compatibility
------------------

Triton-dejavu is currently compatible (and tested) with triton versions 2.2 and newer. Triton-dejavu is compatible with both officially supported triton backends (nvidia and amd).


Versions
---------------

Triton-dejavu roughly follows [Semantic Versioning](https://semver.org), with stable "releases" tagged. Please see [git tags](https://github.com/IBM/triton-dejavu/tags) for a list of older versions. 


However, due to the experimental nature of this repository, internal API compatibility is not always ensured for major _or_ minor version changes. This means that triton-dejavu _can not reuse cache files_ (and cache file structures) that were created with another minor version (i.e. `0.7.X` can't use caches created with `0.5.X`). 


Environment variables
--------------------------

Triton-dejavu can be configured with the following environment variables:

- `TRITON_DEJAVU_STORAGE = <some-path>`: The path to the triton-dejavu storage folder. **It is strongly recommended to set this environment variable**, otherwise `os.getcwd()` will be used. If the cache is stored differently for individual kernels, the `custom_data_storage` parameter for `triton_dejavu.autotuner` could be used.
- `TRITON_PRINT_AUTOTUNING`: Logs the result of the autotune process (as upstream triton).
- `TRITON_DEJAVU_FORCE_FALLBACK = 1`: See [fallback heuristic](#fallback-heuristic).
- `TRITON_DEJAVU_DEBUG = 1`: Prints debug messages.
- `TRITON_DEJAVU_DEBUG_DEBUG = 1`: Prints more debug messages (This will be replaced with logger levels in the future). 
- `TRITON_DEJAVU_USE_ONLY_RESTORED = 1`: Forces the autotuner to just re-evaluate the configurations that were part of the restored autotuner cache in case a new autotuner run (for a new `key`-tuple) is triggered. This could speed up autotuner evaluations by just considering already tried-and-tested configurations out of a bigger configuration space. 
- `TRITON_DEJAVU_USE_ISOLATED_PROCESS = 1`: Runs the benchmark function of a kernel in a separate process. This allows the autotuner to survive crashes of one kernel version during autotuning, e.g. if an illegal memory access happens due to an invalid kernel configuration.
- `_TRITON_DEJAVU_DETERMINED_ROCM_VERSION` and `_TRITON_DEJAVU_DETERMINED_CUDA_VERSION`: These environment variables overwrite the automatic detection of the cuda or rocm by triton-dejavu. It is not recommended to use them besides exceptional scenarios where it is impossible for triton-dejavu to determine the used versions.

