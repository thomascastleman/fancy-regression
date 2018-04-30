
#ifndef _fileio_
#define _fileio_

#include "network.h"

void construct(char * filename, NeuralNetwork * nn);

void serialize(char * filename, NeuralNetwork * nn);

DataSet * readData(char * filename, int size, int usingLetters);

#endif