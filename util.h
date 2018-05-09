
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

void randomize(Matrix * m, float min, float max);

int randInt(int min, int max);

void randomizeNet(NeuralNetwork * n, float wMin, float wMax, float bMin, float bMax);

int * paramCopy(int * params, int size);

void swapPair(DataSet * d, int p1, int p2);

void shuffle(DataSet * d);

#endif