from setuptools import setup, find_packages

setup(
    name="stEEG_decoder",
    version="0.1.0",
    description="Optimized Binary Decoding Pipeline for EEG/MEG",
    author="Philipp Bierwirth",
    author_email="philipp.bierwirth@uni-marburg.de",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "numba",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)