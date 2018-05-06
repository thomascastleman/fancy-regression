#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include "structs.h"
#include "meth.h"

Matrix * dot(Matrix * a, Matrix * b){
	if (a->cols == b->rows){
		Matrix * c = initMatrix(a->rows, b->cols);

		c->at

		return c;

	} else {
		perror("Cannot multiply (meth.c:dot)"), exit(1);
	}
}

Matrix * hadamard(Matrix * a, Matrix * b) {
	// check same dimensions
	if (a->rows == b->rows && a->cols == b->cols) {
		Matrix * c = initMatrix(a->rows, a->cols);

		// multiply elementwise
		for (int i = 0; i < c->rows; i++) {
			for (int j = 0; j < c->cols; j++) {
				c->at[i][j] = a->at[i][j] * b->at[i][j];
			}
		}

		return c;
	} else {
		perror("Cannot hadamard matrices (meth.c:hadamard)"), exit(1);
	}
}

Matrix * transpose(Matrix * m);

Matrix * add(Matrix * a, Matrix * b);

Matrix * scale(float scalar, Matrix * m);

float sigmoid(float x);

float sigmoidPrime(float x);

Matrix * sig(Matrix * m);

Matrix * sigP(Matrix * m);

// initialize a matrix to random values in a range
void randomize(Matrix * m, float min, float max) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->at[i][j] = ((float) rand() / RAND_MAX) * (max - min) + min;
		}
	}
}