
#ifndef _UTIL_INCLUDED_
#define _UTIL_INCLUDED_

#include "structs.h"

float accuracy(NeuralNetwork * n, DataSet * test);

void freeMatrix(Matrix * m);

void freeNetwork(NeuralNetwork * n);

Matrix * forwardPass(NeuralNetwork * n, Matrix * input);

void freeDP(void ** dp, int size);

void zero(NeuralNetwork * n);

void printMatrix(Matrix * m);

void randomize(Matrix * m, float min, float max);

int randInt(int min, int max);

void randomizeNet(NeuralNetwork * n, float wMin, float wMax, float bMin, float bMax);

void printSideBySide(Matrix * a, Matrix * b);

int * paramCopy(int * params, int size);

#endif