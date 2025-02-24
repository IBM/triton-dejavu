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
from triton.runtime.cache import (
    FileCacheManager,
    default_cache_dir,
    default_dump_dir,
    default_override_dir,
)

from .dejavu_utilities import create_dir_if_not_exist_recursive


def set_triton_cache_manager() -> None:
    """Set environment variable to tell Triton to use a
    custom cache manager"""
    manager = "triton_dejavu.cache_manager:CustomCacheManager"
    os.environ["TRITON_CACHE_MANAGER"] = manager


class CustomCacheManager(FileCacheManager):
    """Re-implements Triton's cache manager, to
      1. allow a cache directory for each process
      2. and a different one for each triton-dejavu run.

    For very large search spaces, a large cache directory is sometimes a problem.

    Adapted from: https://github.com/tdoublep/vllm/blob/3307522289fdfefe323b6c00d0db696651989a2f/vllm/triton_utils/custom_cache_manager.py
    """

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            create_dir_if_not_exist_recursive(self.cache_dir)
        elif override:
            self.cache_dir = default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = (
                os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()
            )
            if self.cache_dir:
                run_id = os.getenv("TRITON_DEJAVU_INSTANCE_RUN_ID", "000000-00000")
                self.cache_dir = f"{self.cache_dir}_{os.getpid()}_{run_id}"
                # print(f"setting triton cache dir to {self.cache_dir}")
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                create_dir_if_not_exist_recursive(self.cache_dir)
            else:
                raise RuntimeError("Could not create or locate cache dir")
