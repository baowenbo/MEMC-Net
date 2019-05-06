import os
import torch
import torch.utils.ffi
from torch.utils.ffi import create_extension

strBasepath = os.path.split(os.path.abspath(__file__))[0] + '/'
 
print("strBasepath")
print(strBasepath)
strHeaders = ['src/my_lib.h']
strSources = ['src/my_lib.c']
strDefines = []
strObjects = []

if torch.cuda.is_available() == True:
    strHeaders += ['src/my_lib_cuda.h']#['src/my_lib.h','src/my_lib_cuda.h']
    strSources += ['src/my_lib_cuda.c']# ['src/my_lib.c','src/my_lib_cuda.c']
    strDefines += [('WITH_CUDA', None)]
    strObjects += ['src/my_lib_kernel.o']
# end

objectExtension = torch.utils.ffi.create_extension(
    name='_ext.my_lib',
    headers=strHeaders,
    sources=strSources,
    verbose=True,
    with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines),
    package=False,
    relative_to=strBasepath,
    include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
    define_macros=strDefines,
    extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects]
)
import  shutil
if __name__ == '__main__':

    objectExtension.build()
