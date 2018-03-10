"""Install tensor2tensor."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='tensor2tensor',
    version='1.5.5',
    description='Tensor2Tensor',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/tensor2tensor',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
        'tensor2tensor.data_generators': ['test_data/*'],
        'tensor2tensor.visualization': [
            'attention.js', 'TransformerVisualization.ipynb'
        ],
    },
    scripts=[
        'tensor2tensor/bin/t2t-trainer',
        'tensor2tensor/bin/t2t-datagen',
        'tensor2tensor/bin/t2t-decoder',
        'tensor2tensor/bin/t2t-make-tf-configs',
        'tensor2tensor/bin/t2t-exporter',
        'tensor2tensor/bin/t2t-query-server',
        'tensor2tensor/bin/t2t-insights-server',
        'tensor2tensor/bin/t2t-avg-all',
        'tensor2tensor/bin/t2t-bleu',
        'tensor2tensor/bin/t2t-translate-all',
    ],
    install_requires=[
        'bz2file',
        'flask',
        'future',
        'gevent',
        'google-api-python-client',
        'gunicorn',
        'gym<=0.9.5',  # gym in version 0.9.6 has some temporary issues.
        'numpy',
        'requests',
        'scipy',
        'sympy',
        'six',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.4.1'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.4.1'],
        'tests': ['pytest', 'h5py', 'mock'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning',
)
