mkdir bin
nvcc ./main.cu -DDIM=3 -DMEMO_DIM=1 -o bin/memofactorize3_1 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=4 -DMEMO_DIM=2 -DMEMO_maxElement=15000 -o bin/memofactorize4_2 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=5 -DMEMO_DIM=2 -DMEMO_maxElement=10000 -o bin/memofactorize5_2 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=5 -DMEMO_DIM=3 -o bin/memofactorize5_3 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=6 -DMEMO_DIM=3 -o bin/memofactorize6_3 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=6 -DMEMO_DIM=4 -o bin/memofactorize6_4 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=7 -DMEMO_DIM=4 -o bin/memofactorize7_4 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=8 -DMEMO_DIM=4 -o bin/memofactorize8_4 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=9 -DMEMO_DIM=4 -o bin/memofactorize9_4 -allow-unsupported-compiler -I ./cuda-samples/Common/
nvcc ./main.cu -DDIM=9 -DMEMO_DIM=5 -o bin/memofactorize9_5 -allow-unsupported-compiler -I ./cuda-samples/Common/

./bin/memofactorize3_1 100000
./bin/memofactorize3_1 200000
./bin/memofactorize3_1 300000

./bin/memofactorize4_2 5000
./bin/memofactorize4_2 10000
./bin/memofactorize4_2 15000

./bin/memofactorize5_2 1000
./bin/memofactorize5_2 3000
./bin/memofactorize5_2 5000
./bin/memofactorize5_2 10000

./bin/memofactorize5_3 1000
./bin/memofactorize5_3 3000
./bin/memofactorize5_3 5000

./bin/memofactorize6_3 1000
./bin/memofactorize6_3 3000
./bin/memofactorize6_3 5000

./bin/memofactorize6_4 1000
./bin/memofactorize6_4 2000
./bin/memofactorize6_4 3000

./bin/memofactorize7_4 1000
./bin/memofactorize7_4 1500
./bin/memofactorize7_4 2000

./bin/memofactorize8_4 1000
./bin/memofactorize8_4 1500
./bin/memofactorize8_4 2000

./bin/memofactorize9_4 500
./bin/memofactorize9_4 1000
./bin/memofactorize9_4 1500

./bin/memofactorize9_5 500
./bin/memofactorize9_5 1000
./bin/memofactorize9_5 1500