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

AffineTest: AffineTest.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} AffineTest.cpp mtlib.o firefly.o -o AffineTest

Threshhold: Threshhold.cpp mtlib.o 
	${CC} ${LDFLAGS} Threshhold.cpp mtlib.o -o Threshhold

mbr_tracker: mbr_tracker.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} mbr_tracker.cpp mtlib.o firefly.o -o mbr_tracker

gear_tracker: gear_tracker.cpp mtlib.o
	${CC} ${LDFLAGS} gear_tracker.cpp mtlib.o  -o gear_tracker

filter_test: filter_test.cpp mtlib.o firefly.o
	${CC} ${LDFLAGS} filter_test.cpp mtlib.o firefly.o -o filter_test

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
	rm *.o
	rm ModelsTest
	rm AffineTest
	rm ShrinkTest
	rm GearTracking
	rm Threshhold
	rm mbr_tracker
	rm gear_tracker
	rm filter_test
	rm video_writer_test
	rm live_filter
	rm capture_stills
	rm firefly_test
	rm perspective_test
	rm figures
	rm test

