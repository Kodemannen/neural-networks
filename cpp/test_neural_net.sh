#!/bin/sh

rm outputfile > /dev/null 2>&1;


# needed for getting SFML to display stufff on WSL on Windows:
export DISPLAY=:0


#g++ src/basic_neural_net.cpp \
g++ src/basic_neural_net.cpp src/data_handler.cpp  src/data.cpp  \
    -I include/ \
    -o outputfile -std=c++17 -O2 \
    -larmadillo \
    # -lsfml-graphics -lsfml-window -lsfml-system \
    # -lglut -lGLU -lGL
    # -I /usr/include/eigen3/ \

# run:
./outputfile ;

