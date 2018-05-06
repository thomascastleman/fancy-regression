#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "structs.h"
#include "meth.h"

Matrix * dot(Matrix * a, Matrix * b){
	if (a->cols == b->rows){
		Matrix * c = initMatrix(a->rows, b->cols);
		
		for (int i = 0; i < a->rows; i++){
			for (int j = 0; j < b->cols; j++){
				c->at[i][j] = 0;
				for (int n = 0; n < a->cols; n++){
					c->at[i][j] += a->at[i][n] * b->at[n][j];
				}
			}
		}

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

// activation function
float sigmoid(float x) {
	return (1 / (1 + exp(-x)));
}

// derivative of activation function
float sigmoidPrime(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

// apply activation function element-wise
Matrix * sig(Matrix * m) {
	Matrix * a = initMatrix(m->rows, m->cols);
	for (int i = 0; i < a->rows; i++) {
		for (int j = 0; j < a->cols; j++) {
			a->at[i][j] = sigmoid(m->at[i][j]);
		}
	}
	return a;
}

// apply activation derivative element-wise
Matrix * sigP(Matrix * m) {
	Matrix * a = initMatrix(m->rows, m->cols);
	for (int i = 0; i < a->rows; i++) {
		for (int j = 0; j < a->cols; j++) {
			a->at[i][j] = sigmoidPrime(m->at[i][j]);
		}
	}
	return a;
}

// initialize a matrix to random values in a range
void randomize(Matrix * m, float min, float max) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->at[i][j] = ((float) rand() / RAND_MAX) * (max - min) + min;
		}
	}
}