#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "structs.h"
#include "meth.h"

// multiply two matrices
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

// multiply matrices element-wise
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

// transpose a matrix
Matrix * transpose(Matrix * m){
	Matrix * a = initMatrix(m->cols, m->rows);
	for (int i = 0; i < m->cols; i++){
		for (int j =0; j < m->rows; j++){
			a->at[i][j] = m->at[j][i];
		}
	}
	return a;
}

// add two matrices together
Matrix * add(Matrix * a, Matrix * b){
	if (a->rows == b->rows && a->cols == b->cols) {
		Matrix * c = initMatrix(a->rows, a->cols);

		for (int i = 0; i < a->rows; i++) {
			for (int j = 0; j < a->cols;j++){
				c->at[i][j] = a->at[i][j] + b->at[i][j];
			}
		} 

		return c;
	} else {
		perror("Cannot add (meth.c:add)"), exit(1);
	}
}

// multiply a matrix by a scalar
Matrix * scale(float scalar, Matrix * m) {
	Matrix * a = initMatrix(m->rows, m->cols);
	for (int i = 0; i < a->rows; i++) {
		for (int j = 0; j < a->cols; j++) {
			a->at[i][j] = scalar * m->at[i][j];
		}
	}
	return a;
}

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

// generate random integer
int randInt(int min, int max) {
	return rand() % (max - min) + min;
}

// compute last layer activation with softmax
Matrix * softMax(Matrix * z) {
	Matrix * a = initMatrix(z->rows, 1);
	int sum = 0, i;
	for (i = 0; i < z->rows; i++) {
		sum += exp(z->at[i][0]);
	}

	for (i = 0; i < z->rows; i++) {
		a->at[i][0] = exp(z->at[i][0]) / sum;
	}

	return a;
}