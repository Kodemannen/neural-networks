#include "basic_neural_net.hpp"

neural_net::neural_net(std::vector<int> nodes)
{
    /*  ____            _       
       | __ )  __ _ ___(_) ___  
       |  _ \ / _` / __| |/ __| 
       | |_) | (_| \__ \ | (__  
       |____/ \__,_|___/_|\___| 

        _ __   ___ _   _ _ __ __ _| |  _ __   ___| |_ 
       | '_ \ / _ \ | | | '__/ _` | | | '_ \ / _ \ __|
       | | | |  __/ |_| | | | (_| | | | | | |  __/ |_ 
       |_| |_|\___|\__,_|_|  \__,_|_| |_| |_|\___|\__|
    */                                                                         

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
    //arma::arma_rng::set_seed_random(); 
    arma::arma_rng::set_seed(2); 

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


// returns predictions and loss 
//std::tuple<arma::colvec, double> neural_net::forward(arma::colvec input, arma::colvec target)
std::tuple<arma::colvec, double> neural_net::forward(data input_obj)
{

    /* weight_matrices[0].print(); */
    /* activations[0].print(); */

    // The 0th activation map is the input layer
    activations[0] = input_obj.feature_vec;

    arma::mat W;
    arma::colvec b;
    arma::colvec a_prev;


    for (int i=0; i<n_layers-1; i++)
    {
        a_prev = activations[i];
        W = weight_matrices[i];
        b = biases[i];

        pre_activations[i] = W.t()*a_prev + b;
        // std::cout << pre_activations[i] << std::endl;

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

    double loss = -std::log(activations.back()[input_obj.enum_label]);

    return {activations.back(), loss};      // activations.back() returns the output layer

}
    

void neural_net::backward(data input_obj, double learning_rate)
{


    int L = n_layers -1;    // weight_matrices, biases, pre_activations are L long.
                            // activations is n_layers long, since its first element 
                            // is the input

    arma::colvec target_vec = input_obj.class_vec;

    arma::colvec nabla_zl;
    // Since we are going backwards, we want the first index l to get the last element
    // in weight_matrices, meaning we must start with l=L-1.
    // We must add +1 when we get activations, since it contain one more element, namely
    // the input, which represent the very first activation layer. 


    //-----------------------------------------------------------------------------------------
    // Get gradients:
    //-----------------------------------------------------------------------------------------
    for (int l=L-1; l>=0; l--)      // These are the same values in reverse that l would have
    {                               // if we ran forwards like for (int l=0, l<L; l++) 

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


    
    //-----------------------------------------------------------------------------------------
    // Perform weight update:
    //-----------------------------------------------------------------------------------------
    
    for (int i=0; i<n_layers-1; i++)
    {
        // weight_matrices[i] -= learning_rate * arma::normalise(weight_gradients[i] );
        // biases[i] -= learning_rate * arma::normalise(bias_gradients[i]);
        weight_matrices[i] -= learning_rate * weight_gradients[i] ;
        biases[i] -= learning_rate * bias_gradients[i];
    }
}


// double neural_net::cross_entropy_loss(arma::colvec prediction, int class_index)
// {

// }

void neural_net::train(data_handler dh, int epochs, int mini_batch_size, double learning_rate)
{
    
    std::vector training_data = dh.get_training_data();
    std::vector validation_data = dh.get_validation_data();

    int data_dim = training_data.at(0).feature_vec.size();
    int n_training_data = training_data.size();

    int n_mini_batches = n_training_data / mini_batch_size;
    int residue_batch_size = n_training_data % n_mini_batches;  // The last batch may be smaller 
                                                                // than the mini batch.

    arma::colvec input; 
    arma::colvec target; 

    //this->forward(datapoint_example);


    std::ofstream outfile;
    std::string filename = "output/training_cost.txt";
    outfile.open(filename);

    for (int i=0; i<epochs; i++)
    {

        // std::ofstream outfile;
        // std::string filename = "output/training_cost/epoch-" + std::to_string(i);
        // outfile.open(filename);

        arma::Col order = arma::randperm(n_training_data);

        double cost=0;
        int n_correct = 0;
        int index;
        for (int j=0; j<n_training_data; j++)
        {

            index = order[j];

            if (j % 10000 == 0)
            {
                std::cout << j << std::endl;
            }

            // input = training_data[index].feature_vec;
            // target = training_data[index].class_vec;
            
            data input_obj = training_data[index];

            auto [prediction, loss] = this->forward(input_obj);

            int choice = prediction.index_max();

            if (choice == input_obj.enum_label)
            {
                n_correct += 1;
            }

            
            // Backprop:
            this->backward(input_obj, learning_rate);


            std::cout << weight_matrices[0] << std::endl;


            // Sum up losses into cost:
            cost += loss;


            // this->forward(input, target);
            // arma::colvec prediction = this->get_predictions();


            // // backprop after mini_batch_size forward steps:
            // if ( ((j+1) % (mini_batch_size) == 0) )
            // {

            //     // Take the average:
            //     cost = cost / float(mini_batch_size);
                
            //     // Store cost in file
            //     outfile << cost << std::endl;



            //     cost = 0;

            // } 
        }


        // check accuracy
        std::cout << n_correct << std::endl;
        double acc = float(n_correct) / n_training_data;

        cost = cost / float(n_training_data);
        std::cout << "epoch: " << i << " cost: "<< cost << " acc: " << acc << std::endl;
        outfile << cost << std::endl;

        // // here we do the same with the last residual batch:
        // if (residue_batch_size>0)
        // {
        //     cost = cost / float(residue_batch_size);
        //     outfile << n_training_data << " " << cost << std::endl;
        // }

    }
    outfile.close();
}


//arma::colvec neural_net::get_predictions()
//{
//    return activations[n_layers-1];
//}
