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

from setuptools import setup, Extension
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def read(*path):
    return open(os.path.join(*path), encoding="utf8").read()


def read_requirements(filename):
    return read(PROJECT_ROOT, filename).splitlines()


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/vllm-project/vllm/blob/717f4bcea036a049e86802b3a05dd6f7cd17efc8/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="triton_dejavu",
    version=find_version(os.path.join(PROJECT_ROOT, "triton_dejavu/__init__.py")),
    use_scm_version=True,
    description="Framework to reduce triton overhead to (close to) 0 for well known deployments.",
    long_description=read(PROJECT_ROOT, "README.md"),
    long_description_content_type="text/markdown",
    author="Burkhard Ringlein",
    python_requires=">=3.8",
    packages=["triton_dejavu"],
    install_requires=["triton>2.2", "torch>2.3"],
    extras_require={"BO": ["numpy>=1.23.3", "smac>=2.1.0"]},
)
