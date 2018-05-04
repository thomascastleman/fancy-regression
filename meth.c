
#include "meth.h"

Matrix * dot(Matrix * a, Matrix * b);

Matrix * hadamard(Matrix * a, Matrix * b);

Matrix * transpose(Matrix * m);

Matrix * add(Matrix * a, Matrix * b);

Matrix * scale(float scalar, Matrix * m);

float sigmoid(float x);

float sigmoidPrime(float x);

Matrix * sig(Matrix * m);

Matrix * sigP(Matrix * m);

void randomize(Matrix * m, int min, int max) {
	printf("%d\n", rand() % 20);
	// for (int i = 0; i < m->rows; i++) {
	// 	for (int j = 0; j < m->cols; j++) {

	// 	}
	// }
}