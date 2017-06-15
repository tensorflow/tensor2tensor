"""Install tensor2tensor."""

from setuptools import setup, find_packages

setup(
    name='tensor2tensor',
    version='1.0.1.dev1',
    description='Tensor2Tensor',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/tensor2tensor',
    license='Apache 2.0',
    packages=find_packages(),
    scripts=['tensor2tensor/bin/t2t-trainer', 'tensor2tensor/bin/t2t-datagen'],
    install_requires=[
        'numpy',
        'sympy',
        'six',
        'tensorflow-gpu>=1.2.0rc1',
    ],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow',
)
