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


import sys
import os
import json


__cache_json_template__ = {
    "signature": None,
    "total_bench_time_s": 0.0,
    "evaluated_configs": None,
    "cache": {},
    "timings": {},
}


def print_usage(argv0):
    print(
        f"{argv0} path/to/merge1/ path/to/merge2/cache.json path/to/merge3/tag ... path/to/out/folder/or/file/"
    )


def create_dir_if_not_exist_recursive(path, mode=0o777):
    # 0777 permissions to avoid problems with different users in containers and host system
    norm_path = os.path.normpath(path)
    paths_l = norm_path.split(os.sep)
    path_walked = f"{os.sep}"
    for p in paths_l:
        if len(p) == 0:
            continue
        path_walked = os.path.join(path_walked, p)
        create_dir_if_not_exist(path_walked, mode)


def create_dir_if_not_exist(path, mode=0o777):
    if not os.path.exists(path):
        os.mkdir(path)
        try:
            os.chmod(path, mode)
        except PermissionError as e:
            print(f"can't set permission of directory {path}: {e}")


def merge_cache_files(args):
    merge_soruce = args[:-1]
    merge_target = args[-1]
    if os.path.isdir(merge_target):
        merge_target += "/cache.json"
    target_dir = os.path.dirname(merge_target)
    print(f"Merging dejavu caches {merge_soruce} to {merge_target}...")

    target_cache = __cache_json_template__.copy()
    for sf in merge_soruce:
        if os.path.isdir(sf):
            sf += "/cache.json"
        if not os.path.isfile(sf):
            print(f"can't open {sf}, skipping...")
            continue
        with open(sf, "r") as f:
            cache_json = json.load(f)

        if target_cache["signature"] is None:
            target_cache["signature"] = cache_json["signature"]
        elif target_cache["signature"] != cache_json["signature"]:
            print(f"Signature of {sf} differs, can't merge. STOP")
            return -1
        if target_cache["evaluated_configs"] is None:
            target_cache["evaluated_configs"] = cache_json["evaluated_configs"]
        elif target_cache["evaluated_configs"] != cache_json["evaluated_configs"]:
            print(f"Configspace of {sf} differs, can't merge. STOP")
            return -1

        target_cache["total_bench_time_s"] += cache_json["total_bench_time_s"]

        new_keys = list(cache_json["cache"].keys())
        conflicts = [key for key in new_keys if key in target_cache["cache"]]
        if len(conflicts) > 0:
            print(conflicts)
            print(
                f"Some keys of {sf} did already exist, conflict! STOP (no output created)."
            )
            return -1
        target_cache["cache"].update(cache_json["cache"])
        target_cache["timings"].update(cache_json["timings"])

    create_dir_if_not_exist_recursive(target_dir)
    with open(merge_target, "w") as f:
        json.dump(target_cache, f, indent=4)
    try:
        os.chmod(target_cache, 0o0777)
    except PermissionError as e:
        print(f"can't set permission of file {target_cache}: {e}")
    print(f"...done.")
    return 0


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv or len(sys.argv) < 4:
        print_usage(sys.argv[0])
    else:
        rv = merge_cache_files(sys.argv[1:])
        exit(rv)
