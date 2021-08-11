#include <armadillo>
#include <vector>
#include <map>
#include <string>
#include <array>
#include <cmath>


#include <iostream>
#include <fstream>

#include <stdio.h> // for printf



// random numbers:
#include <cstdlib>      // srand() from here?
#include <ctime>        // and rand() from here?

// for delay:
#include <unistd.h>

#include "data_handler.hpp"

#include "data.hpp"
#include <tuple>

arma::colvec relu(arma::colvec);
arma::colvec relu_derivative(arma::colvec);


arma::colvec softmax(arma::colvec);
arma::colvec softmax_gradient(arma::colvec);

class neural_net
{
    public:
    neural_net(std::vector<int> nodes);
    ~neural_net();


    data current_input_object = data(0);

    //std::tuple<arma::colvec, double> forward(arma::colvec input, arma::colvec target);
    std::tuple<arma::colvec, double> forward(data input_obj);
    void accumulate_gradient();
    void zero_gradient();
    void update_weights(double learning_rate);

    double cross_entropy_loss(arma::colvec prediction, arma::colvec target);
    void train(data_handler dh, int epochs, int mini_batch_size, double learning_rate);

    arma::colvec get_predictions();

    int n_layers;
    std::vector<int> nodes; // nodes in each layer

    double learning_rate;

    std::vector<arma::mat> weight_matrices;
    std::vector<arma::colvec> biases;
    std::vector<arma::colvec> pre_activations;
    std::vector<arma::colvec> activations;

    std::vector<arma::mat> weight_gradients;
    std::vector<arma::colvec> bias_gradients;

};
