"""Install tensor2tensor."""

from distutils.core import setup

setup(
    name='tensor2tensor',
    version='1.0',
    description='Tensor2Tensor',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/tensor2tensor',
    license='Apache 2.0',
    packages=[
        'tensor2tensor', 'tensor2tensor.utils', 'tensor2tensor.data_generators',
        'tensor2tensor.models'
    ],
    scripts=['tensor2tensor/bin/t2t-trainer', 'tensor2tensor/bin/t2t-datagen'],
    install_requires=[
        'numpy',
        'sympy',
        'six',
        'tensorflow-gpu>=1.2.0rc1',
    ],)
