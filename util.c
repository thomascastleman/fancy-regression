#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "structs.h"
#include "meth.h"
#include "util.h"

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

// print vectors side by side (good for comparison)
void printSideBySide(Matrix * a, Matrix * b) {
	for (int i = 0; i < a->rows; i++) {
		printf("%f --> %f\n", a->at[i][0], b->at[i][0]);
	}
}

// pass an input vector through a network
Matrix * forwardPass(NeuralNetwork * n, Matrix * input) {
	int l;
	Matrix *weighted, *plusBias;
	// copy input as first layer activation
	Matrix * act = initMatrix(input->rows, input->cols);
	for (int l = 0; l < act->rows; l++) {
		act->at[l][0] = input->at[l][0];
	}

	// pass activation through network
	for (l = 0; l < n->numberOfLayers - 1; l++) {
		weighted = dot(n->w[l], act);
		plusBias = add(weighted, n->b[l]);

		// activate, apply softmax at last layer
		if (l < n->numberOfLayers - 2)
			act = sig(plusBias);
		else
			act = softMax(plusBias);

		freeMatrix(weighted);
		freeMatrix(plusBias);
	}

	return act;
}

// determine accuracy of classification over a test set
float accuracy(NeuralNetwork * n, DataSet * test) {
	int numCorrect = 0, i, j, max;

	// for every pair in the test set
	for (i = 0; i < test->size; i++) {
		// compute network output
		Matrix * output = forwardPass(n, test->inputs[i]);

		// determine network's most confident classification
		max = 0;
		for (j = 0; j < output->rows; j++) {
			if (output->at[j][0] > output->at[max][0]) {
				max = j;
			}
		}

		if (test->outputs[i]->at[max][0] == 1.0)
			numCorrect++;

		freeMatrix(output);
	}
	return numCorrect / (float) test->size;
}

// free a matrix
void freeMatrix(Matrix * m) {
	// free every row
	for (int i = 0; i < m->rows; i++) {
		free(m->at[i]);
	}
	free(m->at); // free pointer to float[][] array
	free(m); // free pointer to struct
}

// fully free an entire network
void freeNetwork(NeuralNetwork * n) {
	// free all weight matrices / bias vectors
	for (int i = 0; i < n->numberOfLayers - 1; i++) {
		freeMatrix(n->w[i]);
		freeMatrix(n->b[i]);
	}
	free(n->w);	// free pointer to weight matrices
	free(n->b);	// free pointer to bias vectors
	free(n->params); // free pointer to network parameters
	free(n); // free pointer to struct
}

// set entire matrix to 0
void zero(Matrix * m) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->at[i][j] = 0.0f;
		}
	}
}

// initialize a matrix to random values in a range
void randomize(Matrix * m, float min, float max) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->at[i][j] = ((float) rand() / RAND_MAX) * (max - min) + min;
		}
	}
}

// generate random integer in range
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

// copy network parameters
int * paramCopy(int * params, int size) {
	int * copy = malloc(size * sizeof(int));
	for (int i = 0; i < size; i++) {
		copy[i] = params[i];
	}
	return copy;
}

// swap a training pair within a dataset
void swapPair(DataSet * d, int p1, int p2) {
	Matrix * inp = d->inputs[p1];
	Matrix * out = d->outputs[p1];
	d->inputs[p1] = d->inputs[p2];
	d->outputs[p1] = d->outputs[p2];
	d->inputs[p2] = inp;
	d->outputs[p2] = out;
}

// randomize the order of pairs in a dataset
void shuffle(DataSet * d) {
	// swap each pair with a randomly chosen pair
	for (int n = 0; n < d->size; n++) {
		swapPair(d, n, randInt(0, d->size));
	}
}