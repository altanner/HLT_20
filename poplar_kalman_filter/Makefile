#CXX ?= c++

#CXXFLAGS=-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0

#LDFLAGS=-L/opt/poplar/lib64/ -L/opt/boost_libs/

#LIBS=-lboost_timer -lpoplin -lpopops -lpoputil -lpoplar

#all:
	#$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIBS) KalmanFilter.cpp execute-kalman-filter.cpp -o kalman-filter



CXX ?= c++

all:
	$(CXX) -O3 -std=c++17 KalmanFilter.cpp execute-kalman-filter.cpp -lboost_timer -lpoplin -lpopops -lpoputil -lpoplar -o kalman-filter

