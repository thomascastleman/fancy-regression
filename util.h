
#ifndef _UTIL_INCLUDED_
#define _UTIL_INCLUDED_

#include "structs.h"

void printMatrix(Matrix * m);

void printSideBySide(Matrix * a, Matrix * b);

Matrix * forwardPass(NeuralNetwork * n, Matrix * input);

float accuracy(NeuralNetwork * n, DataSet * test);

void freeMatrix(Matrix * m);

void freeNetwork(NeuralNetwork * n);

void zero(Matrix * m);

void randomizeNet(NeuralNetwork * n, float wMin, float wMax, float bMin, float bMax);

int * paramCopy(int * params, int size);

void shuffle(DataSet * d);

#endif