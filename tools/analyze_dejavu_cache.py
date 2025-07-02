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


import sys
import os
import json

from triton_dejavu.dejavu_storage import load_cache_file
from triton_dejavu.dejavu_utilities import get_triton_config_parameter_names


def print_usage(argv0):
    print(
        f"usage:\n{argv0} path/to/dejavu/cache.json\n" +
        f"requirements: triton_dejavu scikit-learn"
    )


def analyze_cache_files(dejavu_cache):
    cache_path = os.path.abspath(dejavu_cache)
    print(f"Analyzing cache in {cache_path}...")
    cache_json, triton_cache, used_configs_list = load_cache_file(cache_path)
    assert len(triton_cache) > 1, "Cache must have at least 2 entires"

    # analyze difference in configs
    # existing_parameters = get_triton_config_parameter_names()
    __non_config_names__ = ["kwargs", "pre_hook", "all_kwargs"]
    existing_parameter_names = [
        s for s in dir(used_configs_list[0]) if s[0:2] != "__" and s not in __non_config_names__
    ]
    existing_kwarg_names = [s for s in list(used_configs_list[0].kwargs.keys()) if s[0:2] != "__" and s not in existing_parameter_names]
    # existing_parameter_names = list(set(existing_parameter_names))
    print(existing_parameter_names, existing_kwarg_names)
    freq_of_parameters = {n:0 for n in existing_parameter_names}
    freq_of_parameters.update({k:0 for k in existing_kwarg_names})
    # print(used_configs_list[0])
    for c in used_configs_list:
        for p in existing_parameter_names:
            v = getattr(c, p)
            if v != 0 and v != None:
                freq_of_parameters[p] += 1
        for k in existing_kwarg_names:
            v = c.kwargs[k]
            if v != 0 and v != None:
                freq_of_parameters[k] += 1
    print(freq_of_parameters)
    used_parameters = {k:v for k,v in freq_of_parameters.items() if v > 0}

    feature_vectors = {}
    class_vectors = {}
    for p in used_parameters.keys():
        feature_vectors_single = []
        class_vectors_single = []
        for k,c in triton_cache.items():
            feature_vectors_single.append(k)
            if p in existing_kwarg_names:
                class_value = c.kwargs[p]
            else:
                class_value = getattr(c, p)
            class_vectors_single.append(class_value)
        # filter classes with only one value
        if len(set(class_vectors_single)) < 2:
            continue
        feature_vectors[p] = feature_vectors_single
        class_vectors[p] = class_vectors_single
    # print(feature_vectors)
    # print(class_vectors)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    analyzed_parameter_names = list(feature_vectors.keys())
    clf = clf.fit(feature_vectors[analyzed_parameter_names[0]], class_vectors[analyzed_parameter_names[0]])
    # tree.plot_tree(clf)
    print(tree.export_text(clf))

    return 0


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv or len(sys.argv) != 2:
        print_usage(sys.argv[0])
    else:
        rv = analyze_cache_files(sys.argv[1])
        exit(rv)
