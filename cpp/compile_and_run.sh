#!/bin/sh

rm outputfile > /dev/null 2>&1;


# needed for getting SFML to display stufff on WSL on Windows:
export DISPLAY=:0


# compile with both Armadillo and Eigen:
#g++  main.cpp src/basicNN.cpp headers/basicNN.h \
#g++  main.cpp basicNN.cpp headers/basicNN.h \
g++ main.cpp src/basic_neural_net.cpp src/data_handler.cpp  src/data.cpp  \
    -I include/ \
    -o writefile -std=c++17 -O2 \
    -larmadillo \
    -lsfml-graphics -lsfml-window -lsfml-system \
    -lglut -lGLU -lGL
    # -I /usr/include/eigen3/ \

# run:
./writefile ;

