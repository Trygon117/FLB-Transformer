from setuptools import setup, find_packages # <--- Add find_packages here
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='flb_transformer', # You can name the overall project here
    packages=find_packages(), # <--- THIS IS THE MAGIC LINE
    ext_modules=[
        CppExtension(
            name='wavefront_backend', 
            sources=['csrc/wavefront_backend.cpp'],
            extra_compile_args=['-O3']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)