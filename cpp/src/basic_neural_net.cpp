#include "basic_neural_net.hpp"

neural_net::neural_net(std::vector<int> nodes, double learning_rate)
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
        W = arma::randn<arma::mat>(nodes[i], nodes[i+1] );  // drawn from gaussian(0,1)
        weight_matrices.push_back(W);

        b = arma::zeros<arma::colvec>(nodes[i+1]);
        biases.push_back(b);

        z = arma::zeros<arma::colvec>(nodes[i+1]);  
        pre_activations.push_back(z);

        a = arma::zeros<arma::colvec>(nodes[i+1]);  
        activations.push_back(a);

        // Gradient placeholders:
        W = arma::zeros<arma::mat>( nodes[i], nodes[i+1] );  // drawn from gaussian(0,1)
        weight_gradients.push_back(W);

        b = arma::zeros<arma::colvec>(nodes[i+1]);
        bias_gradients.push_back(b);
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

arma::colvec relu_derivative(arma::colvec v)
{
    // to get it to return a colvec, we must multiply elementwise with an arma::colvec
    arma::colvec k = arma::ones(v.size());
    return k%(v>0);
}

/* arma::colvec relu_gradient(arma::colvec v) */
/* { */
/*     // H does not become an arma::colvec here, so must use auto. */
/*     // If H is plussed with an arma::colvec, then the result will be arma::colvec */
/*     auto H = v>=0; */
/*     return H; */
/* } */

arma::colvec softmax(arma::colvec v)
{
    // This can be improved by doing operations in log-space
    /* arma::colvec exped = arma::exp(v); */
    /* double denominator = arma::sum(exped); */  
    /* arma::colvec softmaxed = exped / denominator; */

    // Numerically stable version:
    arma::colvec shifted = v - arma::max(v);
    arma::colvec exped = arma::exp(shifted); 

    return exped / arma::sum(exped);
}

/* arma::colvec softmax_gradient(arma::colvec target) */
/* { */

/* } */

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

        pre_activations[i] = W.t()*a_prev + b;

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
    

void neural_net::backward(arma::colvec target_vec)
{

    int L = n_layers -1;    // weight_matrices, biases, pre_activations are L long.
                            // activations is n_layers long, since its first element 
                            // is the input


    arma::colvec nabla_zl;
    // Since we are going backwards, we want the first index l to get the last element
    // in weight_matrices, meaning we must start with l=L-1.
    // We must add +1 when we get activations, since it contain one more element, namely
    // the input, which represent the very first activation layer. 


    //----------------------------------------
    // Get gradients:
    //----------------------------------------
    for (int l=L-1; l>=0; l--)      // These are the same values in reverse that l would have
    {                               // if we ran forwards like for (int l=0, l<L; l++) 
        std::cout << l << std::endl;

        if (l==L-1)
        {
            nabla_zl = activations[l+1] - target_vec;

        } else 
        {
            nabla_zl = relu_derivative(pre_activations[l]) % (weight_matrices[l+1]*nabla_zl);
        }

        weight_gradients[l] = activations[l]*nabla_zl.t();
        bias_gradients[l] = nabla_zl;
    }

    
    //----------------------------------------
    // Perform weight update:
    //----------------------------------------
    for (int i=0; i<n_layers-1; i++)
    {
        weight_matrices[i] -= learning_rate * weight_gradients[i];
        biases[i] -= learning_rate * bias_gradients[i];
    }
}












































/* void neural_net::backward(arma::colvec target_vec) */
/* { */
/*     // Assumes cross entropy loss and relu activation in the hidden layers */
/*     // softmax at output layers */
    
/*     //arma::mat nabla_Wl */
/*     arma::mat nabla_Wl; */
/*     arma::colvec nabla_bl; */
/*     arma::colvec nabla_zl; */ 
/*     arma::mat W; */
    

/*     int L = n_layers-1; */     

/*     // Getting the gradients: */
/*     int l = L; */
/*     for (int i=0; i<L; i++) */
/*     { */

/*         if (l==L) // if output layer: */
/*         { */
/*             nabla_zl = activations[L] - target_vec; // seems to be correct */

/*             W = weight_matrices[l]; // seems to be correct */
/*             std::cout << "----------------W----------------" << std::endl; */
/*             std::cout << W << std::endl; */

/*             /1* nabla_Wl = activations[l-1] * nabla_zl.t(); *1/ */
/*             /1* std::cout << nabla_Wl << std::endl; *1/ */



/*         } else */
/*         { */
/*             auto z = pre_activations[l-1]; */
/*             W = weight_matrices[l]; // seems to be correct */

/*             std::cout << "----------------W----------------" << std::endl; */
/*             std::cout << W << std::endl; */



/*             std::cout << "----------------nabla----------------" << std::endl; */


/*             nabla_zl = relu_derivative(z) % (W*nabla_zl); // seems correct */
/*             nabla_Wl = activations[l-1] * nabla_zl.t(); */
            
/*             std::cout << nabla_Wl << std::endl; */


/*             //nabla_bl = nabla_zl; */



/*             //weight_matrices[l] += nabla_Wl; */
/*             //bias_gradients[l] += nabla_bl; */

/*         } */

/*         std::cout << l << std::endl; */
/*         l -= 1; */
/*     } */
/*     std::cout <<"fiasd"  << std::endl; */
/*     exit(0); */

/*     // Performing the weight/bias update: */
    
/* } */
