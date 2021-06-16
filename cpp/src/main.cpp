#include <iostream>
#include <fstream>

#include <stdio.h> // for printf


#include <armadillo>

//#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <SFML/OpenGL.hpp>

// random numbers:
#include <cstdlib>      // srand() from here?
#include <ctime>        // and rand() from here?

// for delay:
#include <unistd.h>

sf::Uint8* armaMatrixToPixels(arma::mat state);
arma::mat enlargeMatrix(arma::mat matr, int k); 


//        _   _                      _              _       
//       | \ | | ___ _   _ _ __ __ _| |  _ __   ___| |_ ___ 
//       |  \| |/ _ \ | | | '__/ _` | | | '_ \ / _ \ __/ __|
//       | |\  |  __/ |_| | | | (_| | | | | | |  __/ |_\__ \
//       |_| \_|\___|\__,_|_|  \__,_|_| |_| |_|\___|\__|___/
//                                                          


int main () {

    std::cout << "fitttta di" << std::endl;



}





sf::Uint8* armaMatrixToPixels(arma::mat state){ 

    int const H = state.n_rows;
    int const W = state.n_cols;

    // Make a matrix:
    sf::Uint8* pixels = new sf::Uint8[W*H*4];
    int val;

    int ind=0; 
    for (int i=0; i<H; i++) {
        for (int j=0; j<W; j++) {

            val = state(i,j)*200;

            // each pixel is represented by a set of four numbers
            // between 0 and 255
            pixels[ind]   = val;      // R
            pixels[ind+1] = val;      // G
            pixels[ind+2] = val;      // B
            pixels[ind+3] = val;      // a

            ind += 4;
        }
    }

    return pixels;
}






arma::mat enlargeMatrix(arma::mat matr, int k) {
    /*
    Function that maps each pixel in matr to a k \times k submatrix in a new matrix, 
    i.e. makes every pixel into a larger square.
    */
    
    int const H = matr.n_rows;    // height
    int const W = matr.n_cols;    // width

    // initialize new matrix:
    arma::mat enlarged(H*k, W*k); 

    // markers for where we insert values into the new matrix:
    int rowMarker=0;       //  
    int colMarker=0;

    // base for the submatrix we will insert into the new:
    auto base = arma::ones(k,k);

    for (int i=0; i<H; i++) {
        for (int j=0; j<W; j++) {

            auto val = matr(i,j); 

            // insert values:
            enlarged(arma::span(rowMarker,rowMarker+k-1), 
                     arma::span(colMarker,colMarker+k-1)) = base*val;

            colMarker += k;
            colMarker = colMarker % (k*W);

        }
        rowMarker += k;
        rowMarker = rowMarker % (k*H);
    }

    return enlarged;

}
