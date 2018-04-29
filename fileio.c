
#include "fileio.h"
#include "networkc"

// construct a network off of a serialization
void construct(char * filename, NeuralNetwork * nn);

// write weights and biases of a network to given file
void serialize(char * filename, NeuralNetwork * nn);

// read in and format training data
TrainingData * readTraining();