#include <iostream>
#include "data_handler.hpp"
#include "data.hpp"


#include <armadillo>

//#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <SFML/OpenGL.hpp>

// random numbers:
#include <cstdlib>      // srand() from here?
#include <ctime>        // and rand() from here?

// for delay:
#include <unistd.h>

#include "basic_neural_net.hpp"


void visualize_feature_vec(arma::colvec);
void test_data_loader();
void test_nn();
void full_test();


int main()
{

    /*                        _        __ __  
              _ __ ___   __ _(_)_ __  / / \ \
             | '_ ` _ \ / _` | | '_ \| |   | |
             | | | | | | (_| | | | | | |   | |
             |_| |_| |_|\__,_|_|_| |_| |   | |
                                      \_\ /_/ 
    */


    //test_data_loader();
    //test_nn();
    full_test();

    return 0;
}



void visualize_feature_vec(arma::colvec feature_vec) 
{
    int W = 28;
    int H = 28;

    arma::arma_rng::set_seed_random(); 

    int img_size = feature_vec.size();

    // Convert the image data to pixels here:
    sf::Uint8* pixels = new sf::Uint8[img_size*4];

    int val;
    int index = 0;
    for (int i=0; i<img_size; i++)
    {

        //val = int(feature_vec->at(i));
        val = feature_vec[i];
        std::cout << val << std::endl;

        pixels[index] = val;
        pixels[index+1] = val;
        pixels[index+2] = val;
        pixels[index+3] = val;

        index += 4;
    }


    //----------------------------------------
    // Create window instance:
    //----------------------------------------
    sf::RenderWindow window(sf::VideoMode(W, H), "My window");

    glEnable(GL_TEXTURE_2D);
    sf::Texture texture;
    texture.create(W, H);
    sf::Sprite sprite(texture);
    texture.update(pixels);

    window.setTitle("Testing");
    
    // for delay:
    unsigned int delay = 0.1*1e6; // ms

    //window.setActive(true);
    texture.update(pixels);
    window.draw(sprite);
    window.display();

    int count = 0;

    bool running = true;
    while (running) {

        //----------------------------------------
        // Clear and update screen:
        //----------------------------------------
        window.clear(sf::Color::Black);

        texture.update(pixels);
        window.draw(sprite);
        window.display();

        count += 1;
        //std::cout << count << std::endl;
        std::cout << count << std::endl;

        // add some delay:
        usleep(delay);

        if (count >= 30) { 
            break;
        }

    }
}


void test_data_loader()
{

    //std::cout << "successfull compilation compiled successfully" << std::endl; 

    //data_handler dh = data_handler();
    //dh.read_feature_vector("data/mnist/train-images-idx3-ubyte");
    //// dh is a pointer to a c vector containing data objects

    ////auto datap = *dh->at(0);

    //dh.read_feature_labels("data/mnist/train-labels-idx1-ubyte");
    //dh.count_classes();
    //dh.set_labels_properly();  // set enum_label and onehot encoded label 
    //                            // vector for all data
    //dh.split_data();


    //// Training data should be a vector of pointers to data objects
    ////auto training_data = dh->get_training_data();
    
    //// This is how one datapoint is fetched:
    //std::vector<data> training_data = dh.get_training_data();
    //data datapoint = training_data.at(2);


    //arma::colvec inp = datapoint.feature_vec;
    //arma::colvec targ_vec = datapoint.class_vec;


    //int n_inp=inp.size();
    //int n_output=targ_vec.size();

    

    //std::vector<int> nodes = {n_inp, 9, 120, n_output};      

    //double learning_rate=0.001;
    // neural_net nn = neural_net(nodes);
    // nn.forward(inp);
    // nn.backward(targ_vec);

    
    
    //visualize_feature_vec(inp);
}


// void test_nn()
// {

//     double learning_rate=0.001;

//     // Dummy 
//     int n_inp=4;
//     int n_outp=4;
//     std::vector<int> nodes = {n_inp, 20, 30, 4, n_outp};  

//     arma::colvec inp = arma::ones<arma::colvec>(n_inp);

//     arma::colvec targ_vec = arma::zeros<arma::colvec>(n_outp);
//     targ_vec[1] = 1;

//     neural_net nn = neural_net(nodes);

//     nn.forward(inp);
//     nn.backward(targ_vec);
// }

void full_test()
{

    std::cout << "-------------------Full test---------------------" << std::endl;
    //
    //--------------------------------------------------------------------------------------------
    // Load data into the data handler:
    //--------------------------------------------------------------------------------------------
    data_handler dh = data_handler();
    dh.read_feature_vector("data/mnist/train-images-idx3-ubyte");
    dh.read_feature_labels("data/mnist/train-labels-idx1-ubyte");
    dh.count_classes();
    dh.set_labels_properly();  // set enum_label and onehot encoded label vector for all data
    dh.split_data();

    
    // // This is how one datapoint is fetched:
    std::vector<data> training_data = dh.get_training_data();
    data datapoint_example = training_data.at(2);

    arma::colvec inp_example = datapoint_example.feature_vec;
    arma::colvec targ_vec_example = datapoint_example.class_vec;

    int n_inp=inp_example.size();
    int n_output=targ_vec_example.size();

    

    //--------------------------------------------------------------------------------------------
    // Hyper-parameters:
    //--------------------------------------------------------------------------------------------
    std::vector<int> nodes = {n_inp, 2, 3, n_output};      
    int epochs = 1;
    double learning_rate=0.001;
    int mini_batch_size=100;


    //--------------------------------------------------------------------------------------------
    // Initialize neural network:
    //--------------------------------------------------------------------------------------------
    neural_net nn = neural_net(nodes);
    //nn.train(dh, epochs, mini_batch_size, learning_rate);
    auto [prediction, loss] = nn.forward(inp_example, targ_vec_example);
    std::cout << prediction << std::endl;
    std::cout << loss << std::endl;

    // nn.backward(targ_vec);

    std::cout << "nice bruv" << std::endl; 
}
