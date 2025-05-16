mkdir bin
nvcc .\memofactorize.cu -DDIM=8 -DMEMO_DIM=4 -o bin/memofactorize8_4.exe -allow-unsupported-compiler -I ..\..\..\Common\

.\bin\memofactorize8_4.exe