#include <armadillo>
#include <vector>
#include <map>
#include <string>
#include <array>


#include <iostream>
#include <fstream>

#include <stdio.h> // for printf



// random numbers:
#include <cstdlib>      // srand() from here?
#include <ctime>        // and rand() from here?

// for delay:
#include <unistd.h>


arma::colvec relu(arma::colvec);
arma::colvec relu_gradient(arma::colvec);


arma::colvec softmax(arma::colvec);
arma::colvec softmax_gradient(arma::colvec);

class neural_net
{
    private:
    arma::mat W;
    arma::colvec a;

    public:
    neural_net(std::vector<int> );
    ~neural_net();

    void forward(arma::colvec input);
    void backward(arma::colvec target);

    int n_layers;
    std::vector<int> nodes; 
    std::vector<arma::mat> weight_matrices;
    std::vector<arma::colvec> biases;
    std::vector<arma::colvec> pre_activations;
    std::vector<arma::colvec> activations;

};
