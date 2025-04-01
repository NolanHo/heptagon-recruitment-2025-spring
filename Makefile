# CFLAG = -Ofast -g -Wall -fopenmp -march=native -ftree-vectorize -ffast-math -I/home/hj/local/jemalloc/include /home/hj/local/jemalloc/lib/libjemalloc.a
CFLAG = -O3 -g -Wall -fopenmp -march=native -ftree-vectorize -ffast-math -flto -fomit-frame-pointer -funroll-loops

# for debug
# CFLAG = -O1 -g -Wall -fopenmp -march=native -ffast-math -flto -fno-omit-frame-pointer


all:
	g++ driver.cc winograd.cc -std=c++17 ${CFLAG} -o winograd

run:
	./winograd conf/small.conf
	
gemm:
	g++ sgemm.cpp -std=c++17 ${CFLAG} -o sgemm

clean:
	rm -f winograd