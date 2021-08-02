#include "basic_neural_net.hpp"




neural_net::neural_net(std::vector<int> nodes)
{
    // nodes contain the number of nodes for all the layers
    
    std::cout << "fitte" << std::endl;
}


// destructor
neural_net::~neural_net()
{

}











int main()
{
    std::cout << "just_testing" << std::endl;

    // Define neural net architecture:
    std::vector<int> layers = {1,2,3};      // first is input layer, last output layer

    neural_net nn = neural_net(layers);
    return 0;
}
