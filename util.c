#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "structs.h"
#include "meth.h"
#include "util.h"

// pass an input vector through a network
Matrix * forwardPass(NeuralNetwork * n, Matrix * input) {
	Matrix * act = input;
	int l;
	// pass activation through network
	for (l = 0; l < n->numberOfLayers - 2; l++) {
		act = sig(add(dot(n->w[l], act), n->b[l]));
	}
	// apply softmax to last layer
	l = n->numberOfLayers - 2;
	act = softMax(add(dot(n->w[l], act), n->b[l]));

	return act;
}

// free a double pointer
void freeDP(void ** dp, int size) {
	for (int i = 0; i < size; i++) {
		free(dp[i]);
	}
	free(dp);
}

// set entire network to 0
void zero(NeuralNetwork * n) {
	int l, j, k;
	// weights
	for (l = 0; l < n->numberOfLayers - 1; l++) {
		for (j = 0; j < n->w[l]->rows; j++) {
			for (k = 0; k < n->w[l]->cols; k++) {
				n->w[l]->at[j][k] = 0.0f;
			}
		}
	}

	// biases
	for (l = 0; l < n->numberOfLayers - 1; l++) {
		for (j = 0; j < n->b[l]->rows; j++) {
			n->b[l]->at[j][0] = 0.0f;
		}
	}
}

// log a matrix to console
void printMatrix(Matrix * m) {
	for (int r = 0; r < m->rows; r++) {
		for (int c = 0; c < m->cols; c++) {
			printf("%f ", m->at[r][c]);
		}
		printf("\n");
	}
	printf("\n");
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

// randomize weights and biases within ranges
void randomizeNet(NeuralNetwork * n, float wMin, float wMax, float bMin, float bMax) {
	for (int l = 0; l < n->numberOfLayers - 1; l++) {
		randomize(n->w[l], wMin, wMax);
		randomize(n->b[l], bMin, bMax);
	}
}

void printSideBySide(Matrix * a, Matrix * b) {
	for (int i = 0; i < a->rows; i++) {
		printf("%f --> %f\n", a->at[i][0], b->at[i][0]);
	}
}