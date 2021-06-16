#!/bin/sh

rm outputfile > /dev/null 2>&1;


# needed for getting SFML to display stufff on WSL on Windows:
export DISPLAY=:0


# compile with both Armadillo and Eigen:
#g++  main.cpp src/basicNN.cpp headers/basicNN.h \
#g++  main.cpp basicNN.cpp headers/basicNN.h \
g++  main.cpp src/basicNN.cpp  \
    -o outputfile -std=c++17 -O2 \
    -larmadillo \
    -I /usr/include/eigen3/ \
    -lsfml-graphics -lsfml-window -lsfml-system \
    -lglut -lGLU -lGL

# run:
./outputfile ;

