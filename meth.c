
#include "meth.h"

Matrix * dot(Matrix * a, Matrix * b);

Matrix * hadamard(Matrix * a, Matrix * b);

Matrix * transpose(Matrix * m);

Matrix * add(Matrix * a, Matrix * b);

Matrix * scale(float scalar, Matrix * m);

float sigmoid(float x);

float sigmoidPrime(float x);

Matrix * sigmoidMatrix(Matrix * m);

Matrix * sigmoidPrimeMatrix(Matrix * m);