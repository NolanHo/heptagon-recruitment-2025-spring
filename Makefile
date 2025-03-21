CFLAG = -O0 -g -Wall -fopenmp

all:
	g++ driver.cc winograd.cc -std=c++11 ${CFLAG} -o winograd

run:
	./winograd conf/small.conf
	
clean:
	rm -f winograd