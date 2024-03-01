import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name="torchdrug",
    version="1.00",
    description="A PyTorch-based toolkit for developing deep learning models in drug discovery",
    author="",
    packages=find_packages(),
    install_requires=[
        # str(r)
        # for r in pkg_resources.parse_requirements(
        #     open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        # )
    ],
    include_package_data=True,
)