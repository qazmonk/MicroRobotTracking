CC = clang++
LDFLAGS=`pkg-config --cflags --libs opencv` -pthread -ldc1394

mtlib.o : mtlib.h mtlib.cpp
	${CC} ${LDFLAGS} -c mtlib.cpp
firefly.o : firefly.cpp firefly.h
	${CC} ${LDFLAGS} -c firefly.cpp

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

AffineTest: AffineTest.cpp mtlib.h
	${CC} ${LDFLAGS} AffineTest.cpp mtlib.o -o AffineTest

Threshhold: Threshhold.cpp mtlib.o 
	${CC} ${LDFLAGS} Threshhold.cpp mtlib.o -o Threshhold

mbr_tracker: mbr_tracker.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} mbr_tracker.cpp mtlib.o firefly.o -o mbr_tracker

gear_tracker: gear_tracker.cpp mtlib.o
	${CC} ${LDFLAGS} gear_tracker.cpp mtlib.o  -o gear_tracker

all:
	make ModelsTest
	make AffineTest
	make GearTracking
	make Threshhold
	make mbr_tracker
	make gear_tracker
clean:
	rm *.o
	rm ModelsTest
	rm AffineTest
	rm ShrinkTest
	rm GearTracking
	rm Threshholdb
	rm mbr_tracker
	rm gear_tracker
