
#ifndef _TRAIN_INCLUDED_
#define _TRAIN_INCLUDED_

#include "structs.h"

void train(NeuralNetwork * n, DataSet * training, int batchSize, float learningRate);

#endif