#include <iostream>

//#include "headers/basicNN.h"
#include "include/basicNN.h"
//#include <armadillo>

#include <vector>
#include "stdint.h"
#include "stdio.h" 

int main() {

    arma::vec layers = {2, 4, 4,3};

    

    basicNN nn;
    nn.init(layers);



    return 0;
}

