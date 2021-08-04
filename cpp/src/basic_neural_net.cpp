#include "basic_neural_net.hpp"

neural_net::neural_net(std::vector<int> nodes)
{
    /*
     * Constructs a neural net object
     * Contains weights for each layer
     
     * Each layer could be an object

    ----------------------------------------------------------------
    Argument    |   Description
    ----------------------------------------------------------------
    nodes       |   vector containing the number of nodes/neurons in each layer

    */
    

    n_layers = nodes.size();

    //----------------------------------------
    // Create weight matrices:
    //----------------------------------------
    // * Stored in a std::map 
    arma::arma_rng::set_seed_random(); 

    arma::mat W;
    arma::colvec b;
    arma::colvec z;     // prior to activation 
    arma::colvec a;


    // There are always 1 more activation maps than matrices,
    // so we store the 0th first:
    a = arma::zeros<arma::colvec>(nodes[0]);  
    activations.push_back(a);

    for (int i=0; i<n_layers-1; i++)
    {
        W = arma::randn<arma::mat>( nodes[i+1], nodes[i] );  // drawn from gaussian(0,1)
        weight_matrices.push_back(W);

        b = arma::ones<arma::colvec>(nodes[i+1]);
        biases.push_back(b);

        z = arma::zeros<arma::colvec>(nodes[i+1]);  
        pre_activations.push_back(z);

        a = arma::zeros<arma::colvec>(nodes[i+1]);  
        activations.push_back(a);
    }



}



// destructor
neural_net::~neural_net()
{
}


arma::colvec relu(arma::colvec v)
{
    return v%(v>0);
}

arma::colvec softmax(arma::colvec v)
{
    // This can be improved by doing operations in log-space
    arma::colvec exped = arma::exp(v);
    double denominator = arma::sum(exped);  
    arma::colvec softmaxed = exped / denominator;

    return softmaxed;
}

void neural_net::forward(arma::colvec input)
{

    /* weight_matrices[0].print(); */
    /* activations[0].print(); */

    // The 0th activation map is the input layer
    activations[0] = input;

    arma::mat W;
    arma::colvec b;
    arma::colvec a_prev;

    for (int i=0; i<n_layers-1; i++)
    {
        a_prev = activations[i];
        W = weight_matrices[i];
        b = biases[i];

        pre_activations[i] = W*a_prev + b;

        // if not output layer --> using relu activation
        if (i!=n_layers-2)
        {
            activations[i+1] = relu(pre_activations[i]);
        }

        else // if output layer --> using softmax activation
        {
            activations[i+1] = softmax(pre_activations[i]);
        }
    }

}
    

int main()
{
    std::cout << "just_testing" << std::endl;

    // Define neural net architecture:
    int n_inp=4;
    std::vector<int> nodes = {n_inp, 9, 120, 6};      // first is input layer, last output layer
    arma::colvec inp = arma::ones<arma::colvec>(n_inp);

    neural_net nn = neural_net(nodes);
    nn.forward(inp);


    return 0;
}
