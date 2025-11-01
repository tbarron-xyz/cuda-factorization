mkdir bin
nvcc ./main.cu -DDIM=8 -DMEMO_DIM=4 -o bin/memofactorize8_4 -allow-unsupported-compiler -I ./cuda-samples/Common/