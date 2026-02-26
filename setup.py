from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

setup(
    name='reslib',
    version='0.1.1',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            'reslib_cpp',
            ['reslib/cpp/res_moelora.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch',
        'transformers',
        'peft',
    ],
    author="ResLib Team",
    description="High-performance library for Res-MoELoRA adaptation of LLMs",
    python_requires='>=3.8',
)
