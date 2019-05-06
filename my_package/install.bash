#!/usr/bin/env bash
pwd
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
 #nvcc -c -o src/my_lib_kernel.o  src/my_lib_kernel.cu \
#     --gpu-architecture=compute_30 --gpu-code=compute_30  \
#      --compiler-options -fPIC  #-I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

echo "The nvcc tool version:"
nvcc -V | grep "release"
echo "The gpu device capability: "
/usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery | grep "Major/Minor"
echo "Please change the -arch parameter in the following line according to the Major and Minor value of cap."
rm src/my_lib_kernel.o
nvcc -c -o src/my_lib_kernel.o  src/my_lib_kernel.cu --verbose  -g -x cu -Xcompiler -fPIC -arch=sm_61 -lineinfo

python build.py

rm -rf "_ext/my_lib/temp"