CC = clang++
LDFLAGS=`pkg-config --cflags --libs opencv` -pthread

mtlib.o : mtlib.h mtlib.cpp
	${CC} ${LDFLAGS} -c mtlib.cpp


ModelsTest.o: ModelsTest.cpp mtlib.h
	${CC} ${LDFLAGS} -c ModelsTest.cpp

ShrinkTest.o: ShrinkTest.cpp mtlib.h
	${CC} ${LDFLAGS} -c ShrinkTest.cpp

GearTracking.o: GearTracking.cpp mtlib.h
	${CC} ${LDFLAGS} -c GearTracking.cpp

ModelsTest: ModelsTest.o mtlib.o
	${CC} ${LDFLAGS} ModelsTest.o mtlib.o -o ModelsTest

ShrinkTest: ShrinkTest.o mtlib.o
	${CC} ${LDFLAGS} ShrinkTest.o mtlib.o -o ShrinkTest

GearTracking: GearTracking.o mtlib.o
	${CC} ${LDFLAGS} GearTracking.o mtlib.o -o GearTracking

AffineTest: mtlib.o
	${CC} ${LDFLAGS} AffineTest.cpp mtlib.o -o AffineTest


all:
	make ModelsTest
	make ShrinkTest
	make AffineTest
	make GearTracking
clean:
	rm *.o
	rm ModelsTest
	rm AffineTest
	rm ShrinkTest
	rm GearTracking
