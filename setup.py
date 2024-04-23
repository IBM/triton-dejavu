from setuptools import setup, find_packages
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def read(*path):
    return open(os.path.join(*path), encoding='utf8').read()


def read_requirements(filename):
    return read(PROJECT_ROOT, filename).splitlines()


setup(name='triton-dejavu',
      use_scm_version=True,
      description="Framework to try to reduce overhead to (close to) 0 for well known deployments.",
      long_description=read(PROJECT_ROOT, 'README.md'),
      long_description_content_type="text/markdown",
      author="Burkhard Ringlein",
      python_requires='>=3.8',
      packages=find_packages('triton_dejavu'),
      package_dir={'': 'triton_dejavu'},
      include_package_data=True)


