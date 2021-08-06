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



void visualize_feature_vec(arma::colvec);
void test_data_loader();



int main()
{

    test_data_loader();

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

    std::cout << "successfull compilation compiled successfully" << std::endl; 

    data_handler *dh = new data_handler();
    dh->read_feature_vector("data/mnist/train-images-idx3-ubyte");
    // dh is a pointer to a c vector containing data objects

    //auto datap = *dh->at(0);

    dh->read_feature_labels("data/mnist/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    /* int num_classes = dh->get_num_classes(); */
    /* std::cout << num_classes << std::endl; */


    dh->set_labels_properly();


    // Training data should be a vector of pointers to data objects
    //auto training_data = dh->get_training_data();
    
    // This is how one datapoint is fetched:
    std::vector<data *> * training_data = dh->get_training_data();
    data* datapoint = training_data->at(2);

    arma::colvec onehot_example = datapoint->get_class_vec();

    std::cout << onehot_example << std::endl;
    int hmm = datapoint->get_enumerated_label();
    std::cout << hmm << std::endl;

    //std::cout << onehot_example << std::endl;

    //arma::colvec feature_vec_example = datapoint->get_feature_vector(); 
    arma::colvec feature_vec_example = datapoint->feature_vec;

    int data_size = feature_vec_example.size();
    



    
    //visualize_feature_vec(feature_vec_example);
}
