CFLAG = -Ofast -g -Wall -fopenmp -march=native -ftree-vectorize

all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -o winograd

run:
	./winograd conf/small.conf
	
clean:
	rm -f winograd