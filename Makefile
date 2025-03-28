# CFLAG = -Ofast -g -Wall -fopenmp -march=native -ftree-vectorize -ffast-math -I/home/hj/local/jemalloc/include /home/hj/local/jemalloc/lib/libjemalloc.a
CFLAG = -Ofast -g -Wall -fopenmp -march=native -ftree-vectorize -ffast-math
all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -o winograd

run:
	./winograd conf/small.conf
	
clean:
	rm -f winograd