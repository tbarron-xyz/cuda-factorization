mkdir bin
nvcc .\memofactorize.cu -DDIM=3 -DMEMO_DIM=1 -o bin/memofactorize3_1.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=4 -DMEMO_DIM=2 -DMEMO_maxElement=15000 -o bin/memofactorize4_2.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=5 -DMEMO_DIM=2 -DMEMO_maxElement=10000 -o bin/memofactorize5_2.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=5 -DMEMO_DIM=3 -o bin/memofactorize5_3.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=6 -DMEMO_DIM=3 -o bin/memofactorize6_3.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=6 -DMEMO_DIM=4 -o bin/memofactorize6_4.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=7 -DMEMO_DIM=4 -o bin/memofactorize7_4.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=8 -DMEMO_DIM=4 -o bin/memofactorize8_4.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=9 -DMEMO_DIM=4 -o bin/memofactorize9_4.exe -allow-unsupported-compiler -I ..\..\..\Common\
nvcc .\memofactorize.cu -DDIM=9 -DMEMO_DIM=5 -o bin/memofactorize9_5.exe -allow-unsupported-compiler -I ..\..\..\Common\

.\bin\memofactorize3_1.exe 100000
.\bin\memofactorize3_1.exe 200000
.\bin\memofactorize3_1.exe 300000

.\bin\memofactorize4_2.exe 5000
.\bin\memofactorize4_2.exe 10000
.\bin\memofactorize4_2.exe 15000

.\bin\memofactorize5_2.exe 1000
.\bin\memofactorize5_2.exe 3000
.\bin\memofactorize5_2.exe 5000
.\bin\memofactorize5_2.exe 10000

.\bin\memofactorize5_3.exe 1000
.\bin\memofactorize5_3.exe 3000
.\bin\memofactorize5_3.exe 5000

.\bin\memofactorize6_3.exe 1000
.\bin\memofactorize6_3.exe 3000
.\bin\memofactorize6_3.exe 5000

.\bin\memofactorize6_4.exe 1000
.\bin\memofactorize6_4.exe 2000
.\bin\memofactorize6_4.exe 3000

.\bin\memofactorize7_4.exe 1000
.\bin\memofactorize7_4.exe 1500
.\bin\memofactorize7_4.exe 2000

.\bin\memofactorize8_4.exe 1000
.\bin\memofactorize8_4.exe 1500
.\bin\memofactorize8_4.exe 2000

.\bin\memofactorize9_4.exe 500
.\bin\memofactorize9_4.exe 1000
.\bin\memofactorize9_4.exe 1500

.\bin\memofactorize9_5.exe 500
.\bin\memofactorize9_5.exe 1000
.\bin\memofactorize9_5.exe 1500
