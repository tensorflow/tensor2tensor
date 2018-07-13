"""Install tensor2tensor."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='tensor2tensor',
    version='1.6.6',
    description='Tensor2Tensor',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/tensor2tensor',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
        'tensor2tensor.data_generators': ['test_data/*'],
        'tensor2tensor.data_generators.wikisum': ['test_data/*'],
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
        'gym',
        'h5py',
        'numpy',
        'oauth2client',
        'requests',
        'scipy',
        'sympy',
        'six',
        'tqdm',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.6.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.6.0'],
        'tests': [
            'absl-py', 'pytest', 'mock', 'pylint', 'jupyter', 'gsutil'
            # Need atari extras for Travis tests, but because gym is already in
            # install_requires, pip skips the atari extras, so we instead do an
            # explicit pip install gym[atari] for the tests.
            # 'gym[atari]',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    dependency_links=[
        'git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans'
    ],
    keywords='tensorflow machine learning',
)
