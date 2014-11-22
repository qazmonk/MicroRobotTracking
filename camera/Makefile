LDFLAGS=`pkg-config --cflags --libs opencv` -pthread

camera : camera.cpp
	g++ ${LDFLAGS} camera.cpp -ldc1394 -o camera
camera2 : camera2.cpp
	g++ ${LDFLAGS} camera2.cpp -ldc1394 -o camera2
