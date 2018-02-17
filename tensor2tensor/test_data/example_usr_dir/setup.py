# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example setup.py for a t2t_usr_dir launching on Cloud ML Engine.

This is only necessary if you have additional required pip packages for the
import of your usr_dir, and only if you're launching t2t-trainer on Cloud ML
Engine with the --cloud_mlengine flag.

Note that the call to setup uses find_packages() and that the location of this
file is alongside the __init__.py file that imports my_submodule.
"""
from setuptools import find_packages
from setuptools import setup
setup(
    name='DummyUsrDirPackage',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gutenberg',
    ],
)
