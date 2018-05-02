
#ifndef _METH_INCLUDED_
#define _METH_INCLUDED_

Matrix * dot(Matrix * a, Matrix * b);

Matrix * hadamard(Matrix * a, Matrix * b);

Matrix * transpose(Matrix * m);

Matrix * add(Matrix * a, Matrix * b);

Matrix * scale(float scalar, Matrix * m);

Matrix * sigmoidMatrix(Matrix * m);

Matrix * sigmoidPrimeMatrix(Matrix * m);

#endif