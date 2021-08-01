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





/* sf::Uint8* get_pixels() */ 
/* { */
/* }; */






int main()
{
    std::cout << "successfull compilation compiled successfully" << std::endl; 

    data_handler *dh = new data_handler();
    dh->read_feature_vector("data/mnist/train-images-idx3-ubyte");
    // dh is a pointer to a c vector containing data objects

    //auto datap = *dh->at(0);

    dh->read_feature_labels("data/mnist/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();



    // Training data should be a vector of pointers to data objects
    //auto training_data = dh->get_training_data();
    
    // This is how one datapoint is fetched:
    std::vector<data *> * training_data = dh->get_training_data();
    data* datapoint = training_data->at(2);

    //arma::colvec feature_vec_example = datapoint->get_feature_vector(); 
    arma::colvec feature_vec_example = datapoint->feature_vec;

    int data_dim = feature_vec_example.size();
    
    std::cout << data_dim << std::endl;
    exit(1);

    /* std::vector<uint8_t> * feature_vec = datapoint->get_feature_vector(); */


    /* int data_size = feature_vec->size(); */

    /* //---------------------------------------- */
    /* // */  
    /* //---------------------------------------- */
    /* arma::Col feature_vec_arma = arma::Col(*feature_vec); */

    /* std::cout << feature_vec_arma[0] << std::endl; */
    
    /* std::cout << "asdasi" << std::endl; */
    /* exit(1); */


    /* int const H = 28; */
    /* int const W = 28; */ 


    /* std::cout << "asd" << std::endl; */
    /* // Convert the image data to pixels here: */
    /* sf::Uint8* pixels = new sf::Uint8[W*H*4]; */

    

    /* int val; */
    /* int index = 0; */
    /* for (int i=0; i<H*W; i++) */
    /* { */

    /*     val = int(feature_vec->at(i)); */

    /*     pixels[index] = val; */
    /*     pixels[index+1] = val; */
    /*     pixels[index+2] = val; */
    /*     pixels[index+3] = val; */

    /*     index += 4; */
    /* } */


    /* arma::arma_rng::set_seed_random(); */ 


 
    /* //---------------------------------------- */
    /* // Create window instance: */
    /* //---------------------------------------- */
    /* sf::RenderWindow window(sf::VideoMode(W, H), "My window"); */

    /* //sf::RenderWindow window; */
    /* // can take a third argument: sf::Style::Fullscreen */
    /* // or sf::Style::Resize, and more. */

    /* glEnable(GL_TEXTURE_2D); */


    /* sf::Texture texture; */
    /* texture.create(W, H); */

    /* sf::Sprite sprite(texture); */
    /* texture.update(pixels); */


    /* window.setTitle("Testing"); */
    
    /* // get the size of the window */
    /* //sf::Vector2u size = window.getSize(); */
    /* //unsigned int width = size.x; */
    /* //unsigned int height = size.y; */


    /* // for delay: */
    /* unsigned int delay = 0.1*1e6; // ms */


    /* //window.setActive(true); */
    /* texture.update(pixels); */
    /* window.draw(sprite); */
    /* window.display(); */




    /* int count = 0; */

    /* bool running = true; */
    /* while (running) { */


        

    /*     //---------------------------------------- */
    /*     // Clear and update screen: */
    /*     //---------------------------------------- */
    /*     window.clear(sf::Color::Black); */

    /*     texture.update(pixels); */
    /*     window.draw(sprite); */
    /*     window.display(); */


    /*     count += 1; */
    /*     //std::cout << count << std::endl; */
    /*     std::cout << count << std::endl; */

    /*     // add some delay: */
    /*     usleep(delay); */


    /*     if (count >= 1000) { */ 
    /*         return 0; */
    /*     } */

    /* } */

    return 0;
}


