import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR=os.path.dirname(os.path.abspath(__file__))

cuda_source_dir= os.path.join("teletron","core","cuda","fused_rmsnorm")

def get_cuda_path(filename):
    return os.path.join(cuda_source_dir,filename)

setup(
    
    name="custom_rmsnorm",  
    
    ext_modules=[
        CUDAExtension(
            
            name="custom_rmsnorm",  
            sources=[
                get_cuda_path("rms_ops.cpp"),  # 
                get_cuda_path("rms_forward.cu"),  
                get_cuda_path("rms_backward.cu"),
                get_cuda_path("weight_backward.cu")
            ],
            include_dirs=[os.path.join(ROOT_DIR,get_cuda_path("include"))], 
            extra_compile_args={"cxx": ["-O3"], 
                                'nvcc': [
                    '-O3', 
                    '-arch=sm_90',       
                    '-DENABLE_BF16',     
                    '--use_fast_math',    
                    '-gencode=arch=compute_90,code=sm_90'
                ]
            }  
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)