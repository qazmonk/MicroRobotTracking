CC = clang++
LDFLAGS=`pkg-config --cflags --libs opencv` -pthread

mtlib.o : mtlib.h mtlib.cpp
	${CC} ${LDFLAGS} -c mtlib.cpp


ModelsTest.o: ModelsTest.cpp mtlib.h
	${CC} ${LDFLAGS} -c ModelsTest.cpp

ModelsTest: ModelsTest.o mtlib.o
	${CC} ${LDFLAGS} ModelsTest.o mtlib.o -o ModelsTest

AffineTest: mtlib.o
	${CC} ${LDFLAGS} AffineTest.cpp mtlib.o -o AffineTest

all:
	make ModelsTest

clean:
	rm *.o
	rm ModelsTest
