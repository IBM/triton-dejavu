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

##########################################
# Some utilities for working with triton #
##########################################

from __future__ import annotations

import builtins
import sys
import os
import time
import random
import string


# from https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/matmul.py (also apache 2.0)
def unpack_grid(grid):
    if len(grid) == 1:
        return grid[0], 1, 1
    if len(grid) == 2:
        return grid[0], grid[1], 1
    if len(grid) == 3:
        return grid[0], grid[1], grid[2]


def get_random_key(prefix=""):
    # is always with replacement
    x = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    if len(prefix) > 0:
        return f"{prefix}-{x}"
    return x


global_metadata_store = {
    "_initialized": f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
    "_pid": f"{os.getpid()}",
}
