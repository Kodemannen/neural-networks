#!/bin/sh


#source ~/repos/useful-stuff/dotfiles/bashaliases.sh

export DISPLAY=:0       # needed on WSL

echo "----------------------------------------"
echo "deleting old files:"
echo "  "
rm outputfile > /dev/null 2>&1

echo "----------------------------------------"
echo "compiling:"
echo "  "
#g++ main.cpp -o writefile 

# Compile with armadillo:
g++ src/main.cpp -o outputfile -std=c++17 -O2 \
    -larmadillo \
    -lsfml-graphics -lsfml-window -lsfml-system \
    -lglut -lGLU -lGL
#      -I /usr/include/eigen3/ \        # if with eigen




#g++ main.cpp -o writefile
echo "----------------------------------------"
echo "output:"
echo "  "
./outputfile 

# plot in python:
#python3 plot.py
#python3 animate.py
