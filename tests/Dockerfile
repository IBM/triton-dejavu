
# FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev
# FROM nvidia/cuda:12.3.0-devel-ubuntu22.04 AS dev
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
# RUN ldconfig /usr/local/cuda-12.1/compat/

WORKDIR /workspace
COPY tests/requirements.txt requirements.txt
RUN pip install -r requirements.txt 

# broken?
# RUN pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
RUN git clone https://github.com/triton-lang/triton.git 
RUN pip install ./triton/python

#### triton dejavu #### 
# RUN pip install -e ./triton-dejavu/

WORKDIR /workspace/triton-dejavu
COPY setup.py setup.py
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
# ENV CONTAINER_CUDA_VERSION=12.1
# ENV CONTAINER_CUDA_VERSION=12.3
ENV CONTAINER_CUDA_VERSION=12.4

# expose debugpy port
EXPOSE 5678

ENTRYPOINT ["python3", "-m", "tests.test_dejavu"]

