
#include <iostream>             // cout, endl 

#include <vector>
#include "stdint.h"
#include "stdio.h" 
#include "data_handler.hpp"

// constructor:
data_handler::data_handler()
{
    // data_array = std::vector<data>;
    // test_data = std::vector<data>;
    // training_data = std::vector<data>;
    // validation_data = std::vector<data>;
};


// destructor:
data_handler::~data_handler()
{
};


void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4]; // |MAGIC|NUM IMGS|ROWSIZE|COLSIZE|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        for (int i=0; i<4; i++)
        {

            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }

        printf("Done getting input file header.\n");
        int image_size = header[2]*header[3];


        // iterate over all imgs and read their data:
        for (int i=0; i<header[1]; i++)
        {
            // make (pointer to) data oject for each image:
            data d = data(image_size);
            uint8_t element[1];

            // iterate over all pixels?
            for (int j=0; j<image_size; j++)
            {
                //
                // Files automatically go to next element after reading one, so by 
                // looping and reading a single byte from f into element, the next 
                // time we do it, we get the next byte in line
                if (fread(element, sizeof(element), 1, f))
                {
                    //d->append_to_feature_vector(element[0]);
                    d.feature_vec[j] = element[0];

                    //printf("$d\n", i);
                    //std::cout << i << std::endl;
                } else
                {
                    printf("Error reading from file \n");
                    exit(1);
                }
            }
            data_array.push_back(d);
        }
        n_total_data = data_array.size();
        printf("Successfully read and stored %lu feature vectors\n", n_total_data);
    } else
    {
        printf("Could not find file \n");
        exit(1);
    }
}


void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2]; // |MAGIC|NUM IMGS|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");

    if (f)
    {
        for (int i=0; i<2; i++)
        {

            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting label file header.\n");
        for (int i=0; i<header[1]; i++)
        {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f))
            {
                data_array.at(i).set_label(element[0]);
            } else
            {
                printf("Error reading from file \n");
                exit(1);
            }
        }
        printf("Successfully read and stored labels\n");
    } else
    {
        printf("Could not find file \n");
        exit(1);
    }
}

void data_handler::split_data()
{
    std::unordered_set<int> used_indexes;
    int train_size = data_array.size() * TRAIN_SET_PERCENT;
    int test_size = data_array.size() * TEST_SET_PERCENT;
    int valid_size = data_array.size() * VALIDATION_SET_PERCENT;

    // Training data:
    int count = 0;
    while (count < train_size)
    {
        int rand_index = rand() % data_array.size(); 
        if (used_indexes.find(rand_index) == used_indexes.end())
        {   // this if test checks if 
            training_data.push_back(data_array.at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }


    // Test data:
    count = 0;
    while (count < test_size)
    {
        int rand_index = rand() % data_array.size(); 
        if (used_indexes.find(rand_index) == used_indexes.end())
        {   // this if test checks if 
            test_data.push_back(data_array.at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    // Validation data:
    count = 0;
    while (count < valid_size)
    {
        int rand_index = rand() % data_array.size(); 
        if (used_indexes.find(rand_index) == used_indexes.end())
        {   // this if test checks if 
            validation_data.push_back(data_array.at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }
    printf("Training data size: %lu \n", training_data.size());
    printf("Test data size: %lu \n", test_data.size());
    printf("Validation data size: %lu \n", validation_data.size());
}


void data_handler::count_classes()
{
    int count = 0;
    for (unsigned i = 0; i<data_array.size(); i++)
    {
        // We iterate over all the data to fill class map
        if (class_map.find(data_array.at(i).get_label()) == class_map.end())
        {
            class_map[data_array.at(i).get_label()] = count;
            count++;
        }
    }
    num_classes = count;
    printf("Successfully extracted %d unique classes \n", num_classes);
}

void data_handler::set_labels_properly()
{
    // set enum_label and onehot vector for all data 

    uint8_t label;
    int enum_label;
    for (unsigned i = 0; i<n_total_data; i++)
    {
        label = data_array.at(i).get_label();
        enum_label = class_map[ label ];

        data_array.at(i).set_enumerated_label(enum_label);
        data_array.at(i).set_class_vec(enum_label, num_classes);
    }
}




uint32_t data_handler::convert_to_little_endian(const unsigned char * bytes)
{
    return (uint32_t) ((bytes[0] << 24)  |
                        (bytes[1] << 16) |
                        (bytes[2] << 8)  |
                        (bytes[3]));
}

uint8_t data_handler::get_num_classes()
{
    return num_classes;
}

std::vector<data> data_handler::get_training_data()
{
    return training_data;
}

std::vector<data> data_handler::get_test_data()
{
    return test_data;
}

std::vector<data> data_handler::get_validation_data()
{
    return validation_data;
}


void data_handler::create_dummy_data(int n_classes, int n_data_per_class, int input_size)
{
    //
    double f0 = 2;
    double T = 1;
    double dt = T / (n_data_per_class-1);
    arma::colvec t_vec = arma::linspace(0, T, input_size);

    double pi = 3.14159265359;
    
    int n_total_data = n_classes*n_data_per_class;
    double noise_var = 0.9;

    for (int i=0; i<n_total_data; i++)
    {

        
        // pick a random class for the data:
        int class_choice = rand() % n_classes;
        double freq = (class_choice+1) * f0;
        double phase = 2*pi / (rand()%1000 + 1);

        data d = data(input_size);

        d.feature_vec = arma::sin(2*pi*freq*t_vec) + phase;
        d.feature_vec += arma::randn(input_size) * noise_var;

        d.set_enumerated_label(class_choice);

        arma::colvec onehot_class_vec = arma::zeros(n_classes);
        onehot_class_vec[class_choice] = 1;
        d.class_vec = onehot_class_vec;

        data_array.push_back(d);
    }

    std::cout << "Created dummy data" << std::endl;
}


    // data_array = std::vector<data>;
    // test_data = std::vector<data>;
    // training_data = std::vector<data>;
    // validation_data = std::vector<data>;


