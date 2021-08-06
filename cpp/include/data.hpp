#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"
#include <ostream>
#include <armadillo>

// Each datapoint is represented as a unique instance of this data class
// if I understand it correctly
class data 
{
    // change this to an armadillo vector?
    //std::vector<uint8_t> * feature_vector;  // the actual input



    public:
    uint8_t label;  // class label of some type, e.g. string "cat"
    int enum_label; // class label represented as integer

    data(int image_size);
    ~data();

    arma::colvec feature_vec;
    arma::colvec class_vec;         // onehot

    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enumerated_label(int);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumerated_label();

    void set_class_vec(int enum_label, int n_classes);
    arma::colvec get_class_vec();

    std::vector<uint8_t> * get_feature_vector();
};
    



#endif
