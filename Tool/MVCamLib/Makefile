PYTHON_INC = -I/usr/include/python2.7
BOOST_INC = -I/usr/local/include

BOOST_LIB=-L/usr/local/lib

OPENCV_LIB=$$(pkg-config --libs opencv)
OPENCV_INC=$$(pkg-config --cflags opencv)
LIB = -L/usr/local/lib $(OPENCV_LIB) $(OPENCV_INC) -lboost_python -L/usr/lib

MVCam_INC = -I./include
MVCam_LIB = -lMVSDK
MVCam_py.so:MVCam_py.o Camera.o conversion.o
	g++ -shared -fpic -Wl,--export-dynamic \
		 MVCam_py.o Camera.o conversion.o  $(PYTHON_INC) $(LIB) $(MVCam_LIB) -o MVCam_py.so

MVCam_py.o:MVCam_py.cpp
	g++ -fpic -c MVCam_py.cpp -o MVCam_py.o $(PYTHON_INC) $(OPENCV_INC)$(MVCam_INC) $(MVCam_LIB)



Camera.o:Camera.cpp Camera.h
	g++ -c -fpic Camera.cpp -o Camera.o $(PYTHON_INC) $(OPENCV_INC) $(MVCam_INC) $(MVCam_LIB) 

conversion.o:conversion.cpp conversion.h
	g++ -c -fpic conversion.cpp -o conversion.o $(PYTHON_INC) 




clean:
	rm *.so *.o 
