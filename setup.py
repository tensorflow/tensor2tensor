"""Install tensor2tensor."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='tensor2tensor',
    version='1.15.4',
    description='Tensor2Tensor',
    long_description=(
        'Tensor2Tensor, or T2T for short, is a library of '
        'deep learning models and datasets designed to make deep '
        'learning more accessible and accelerate ML research. '
        'T2T was developed by researchers and engineers in the Google '
        'Brain team and a community of users. It is now in maintenance '
        'mode -- we keep it running and welcome bug-fixes, but encourage '
        'users to use the successor library Trax.'),
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
        'tensor2tensor/bin/t2t-eval',
        'tensor2tensor/bin/t2t-exporter',
        'tensor2tensor/bin/t2t-query-server',
        'tensor2tensor/bin/t2t-insights-server',
        'tensor2tensor/bin/t2t-avg-all',
        'tensor2tensor/bin/t2t-bleu',
        'tensor2tensor/bin/t2t-translate-all',
    ],
    install_requires=[
        'absl-py',
        'bz2file',
        'dopamine-rl',
        'flask',
        'future',
        'gevent',
        'gin-config',
        'google-api-python-client',
        'gunicorn',
        'gym',
        'h5py',
        'kfac',
        'mesh-tensorflow',
        'numpy',
        'oauth2client',
        'opencv-python',
        'Pillow',
        'pypng',
        'requests',
        'scipy',
        'six',
        'sympy',
        'tensorflow-datasets',
        'tensorflow-gan',
        'tensorflow-probability==0.7.0',
        'tqdm',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.15.0'],
        'tensorflow-hub': ['tensorflow-hub>=0.1.1'],
        'tests': [
            # Needed to fix a Travis pytest error.
            # https://github.com/Julian/jsonschema/issues/449#issuecomment-411406525
            'attrs>=17.4.0',
            'pytest>=3.8.0',
            'mock',
            'jupyter',
            'matplotlib',
            # Need atari extras for Travis tests, but because gym is already in
            # install_requires, pip skips the atari extras, so we instead do an
            # explicit pip install gym[atari] for the tests.
            # 'gym[atari]',
        ],
        'allen': ['Pillow==5.1.0', 'pandas==0.23.0'],
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
