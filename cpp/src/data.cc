#include "data.hpp"

data::data(int image_size)
{
    //feature_vector = new std::vector<uint8_t>;
    // we use new to allocate memory on the heap so that it lasts outside of this scope
    // I think
    feature_vec = arma::zeros(image_size);
    //feature_vec = mat();
}
data::~data()
{

}


/* void data::set_feature_vector(std::vector<uint8_t> * vect) */
/* { */
/*     feature_vector = vect; */
/* } */

void data::append_to_feature_vector(uint8_t val)
{
    //feature_vector->push_back(val);
}

void data::set_label(uint8_t val)
{
    label = val;
}

void data::set_enumerated_label(int val)
{
    enum_label = val;
}

/* int data::get_feature_vector_size() */
/* { */
/*     return feature_vector->size(); */
/* } */

uint8_t data::get_label()
{
    return label;
}

uint8_t data::get_enumerated_label()
{
    return enum_label;
}

//std::vector<uint8_t> * data::get_feature_vector()
/* arma::vec data::get_feature_vector() */
/* { */
/*     return feature_vec; */
/* } */
