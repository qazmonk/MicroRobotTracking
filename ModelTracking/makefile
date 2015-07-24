CC = clang++
CFLAGS = -pthread -Wall -O3
CFLAGS_DEBUG = -pthread -ldc1394 -Wall -O3 -g
LDFLAGS=`pkg-config --cflags --libs opencv` -ldc1394 -Wall -O3
LDFLAGS_DEBUG=`pkg-config --cflags --libs opencv` -ldc1394 -O3 -g -Wall

mtlib.o : mtlib.h mtlib.cpp
	${CC} ${CFLAGS} -c mtlib.cpp

mtlib_debug.o : mtlib.h mtlib.cpp
	${CC} ${CFLAGS_DEBUG} -c mtlib.cpp -o mtlib_debug.o

firefly.o : firefly.cpp firefly.h
	${CC} ${CFLAGS} -c firefly.cpp

firefly_debug.o : firefly.cpp firefly.h
	${CC} ${CFLAGS_DEBUG} -c firefly.cpp -o firefly_debug.o

ModelsTest.o: ModelsTest.cpp mtlib.h
	${CC} ${CFLAGS} -c ModelsTest.cpp

ShrinkTest.o: ShrinkTest.cpp mtlib.h
	${CC} ${CFLAGS} -c ShrinkTest.cpp

GearTracking.o: GearTracking.cpp mtlib.h
	${CC} ${CFLAGS} -c GearTracking.cpp

ModelsTest: ModelsTest.o mtlib.o
	${CC} ${LDFLAGS} ModelsTest.o mtlib.o -o ModelsTest

ShrinkTest: ShrinkTest.o mtlib.o
	${CC} ${LDFLAGS} ShrinkTest.o mtlib.o -o ShrinkTest

GearTracking: GearTracking.o mtlib.o
	${CC} ${LDFLAGS} GearTracking.o mtlib.o -o GearTracking

AffineTest: AffineTest.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} AffineTest.cpp mtlib.o firefly.o -o AffineTest

Threshhold: Threshhold.cpp mtlib.o 
	${CC} ${LDFLAGS} Threshhold.cpp mtlib.o -o Threshhold

mbr_tracker: mbr_tracker.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} -pthread mbr_tracker.cpp mtlib.o firefly.o -o mbr_tracker

gear_tracker: gear_tracker.cpp mtlib.o
	${CC} ${LDFLAGS} gear_tracker.cpp mtlib.o  -o gear_tracker

filter_test: filter_test.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} filter_test.cpp mtlib.o firefly.o -o filter_test

filter_test_debug: filter_test.cpp mtlib_debug.o
	${CC} ${LDFLAGS_DEBUG} filter_test.cpp mtlib_debug.o -o filter_test_debug

video_writer_test: video_writer_test.cpp mtlib.o
	${CC} ${LDFLAGS} video_writer_test.cpp mtlib.o  -o video_writer_test

live_filter: live_filter.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} live_filter.cpp mtlib.o firefly.o -o live_filter

capture_stills: capture_stills.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} capture_stills.cpp mtlib.o firefly.o -o capture_stills

firefly_test : firefly_test.cpp firefly.o mtlib.o
	${CC} ${LDFLAGS} firefly_test.cpp firefly.o mtlib.o -o firefly_test

perspective_test : perspective_test.cpp mtlib.o
	${CC} ${LDFLAGS} perspective_test.cpp mtlib.o -o perspective_test

figures: figures.cpp mtlib.o
	${CC} ${LDFLAGS} figures.cpp mtlib.o -o figures

test: test.cpp mtlib.o
	${CC} ${LDFLAGS} test.cpp mtlib.o -o test

all:
	make ModelsTest
	make AffineTest
	make GearTracking
	make Threshhold
	make mbr_tracker
	make gear_tracker
	make filter_test
	make video_writer_test
	make live_filter
	make capture_stills
	make firefly_test
	make perspective_test
	make figures
	make test
clean:
	rm -f *.o
	rm -f ModelsTest
	rm -f AffineTest
	rm -f ShrinkTest
	rm -f GearTracking
	rm -f Threshhold
	rm -f mbr_tracker
	rm -f gear_tracker
	rm -f filter_test
	rm -f video_writer_test
	rm -f live_filter
	rm -f capture_stills
	rm -f firefly_test
	rm -f perspective_test
	rm -f figures
	rm -f test
	rm -f filter_test_debug

debug:
	make filter_test_debug
