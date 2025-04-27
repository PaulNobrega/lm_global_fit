from setuptools import setup, find_packages

setup(
    name="lm_global_fit",
    version="1.0.0",
    author="R. Paul Nobrega",
    author_email="Paul@PaulNobrega.net",
    description="Levenberg-Marquardt Global Fitter for Python",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PaulNobrega/lm_global_fit",
    packages=find_packages(),
    py_modules=["global_fit"],
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
)
