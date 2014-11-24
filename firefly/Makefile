LDFLAGS=`pkg-config --cflags --libs opencv` -pthread

firefly.o : firefly.cpp firefly.h
	g++ ${LDFLAGS} -c firefly.cpp -ldc1394

firefly_test : firefly_test.cpp firefly.o
	g++ ${LDFLAGS} firefly_test.cpp firefly.o -o firefly_test -ldc1394

