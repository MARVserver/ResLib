from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='reslib_cpp',
    ext_modules=[
        CppExtension(
            'reslib_cpp',
            ['reslib/cpp/res_moelora.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
