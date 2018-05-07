
#ifndef _UTIL_INCLUDED_
#define _UTIL_INCLUDED_

#include "structs.h"

Matrix * forwardPass(NeuralNetwork * n, Matrix * input);

void freeDP(void ** dp, int size);

void zero(NeuralNetwork * n);

void printMatrix(Matrix * m);

void randomize(Matrix * m, float min, float max);

int randInt(int min, int max);

void randomizeNet(NeuralNetwork * n, float wMin, float wMax, float bMin, float bMax);

void printSideBySide(Matrix * a, Matrix * b);

#endif