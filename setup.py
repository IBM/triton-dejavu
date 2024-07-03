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

from setuptools import setup
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def read(*path):
    return open(os.path.join(*path), encoding='utf8').read()


def read_requirements(filename):
    return read(PROJECT_ROOT, filename).splitlines()


setup(name='triton_dejavu',
      use_scm_version=True,
      description="Framework to try to reduce triton overhead to (close to) 0 for well known deployments.",
      # long_description=read(PROJECT_ROOT, 'README.md'),
      long_description_content_type="text/markdown",
      author="Burkhard Ringlein",
      python_requires='>=3.8',
      packages=['triton_dejavu'],
      # package_dir={'': 'triton_dejavu'},
      include_package_data=True)


