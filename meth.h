
#ifndef _METH_INCLUDED_
#define _METH_INCLUDED_

#include "structs.h"

Matrix * dot(Matrix * a, Matrix * b);

Matrix * hadamard(Matrix * a, Matrix * b);

Matrix * transpose(Matrix * m);

Matrix * add(Matrix * a, Matrix * b);

Matrix * scale(float scalar, Matrix * m);

Matrix * sig(Matrix * m);

Matrix * sigP(Matrix * m);

Matrix * softMax(Matrix * z);

#endif