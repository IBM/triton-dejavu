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

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

WORKDIR /workspace
COPY tests/requirements.txt requirements.txt
RUN pip install -r requirements.txt 

# broken?
# RUN pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# ARG triton_version=main
ARG triton_version=release/3.2.x
# main = nightly, basically
RUN git clone https://github.com/triton-lang/triton.git
RUN cd triton; git checkout ${triton_version}
RUN pip install ./triton/python

#### triton dejavu #### 
# RUN pip install -e ./triton-dejavu/

WORKDIR /workspace/triton-dejavu
COPY setup.py setup.py
COPY README.md README.md
COPY triton_dejavu triton_dejavu
RUN pip install ./

#### tests ####

WORKDIR /workspace
COPY tests tests

# for in container development
# RUN chmod a+rwx -R .


# environment
ENV PYTHONUNBUFFERED=1
ENV TRITON_PRINT_AUTOTUNING=1
ENV TRITON_DEJAVU_DEBUG=1
ENV TRITON_DEJAVU_STORAGE=/storage/dejavu-data/

# expose debugpy port
# EXPOSE 5678

ENTRYPOINT ["python3", "-m", "tests.test_dejavu"]

