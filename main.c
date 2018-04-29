#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int rows;
	int cols;
	float ** at;
} Matrix;

typedef struct NN {
	int numberOfLayers;
	int * params;
	Matrix ** w;
	Matrix ** b;
} NeuralNetwork;

// construct a new r by c matrix
Matrix * initMatrix(int r, int c) {
	Matrix * m = malloc(sizeof(Matrix));
	m->rows = r; m->cols = c;
	m->at = (float **) malloc(r * sizeof(float *));
	for (int i = 0; i < r; i++)
		m->at[i] = (float *) malloc(c * sizeof(float));
	return m;
}

// construct a new network off of given params
NeuralNetwork * initNN(int numLayers, int * params) {
	NeuralNetwork * n = malloc(sizeof(NeuralNetwork));
	n->params = params;
	n->numberOfLayers = numLayers;
	n->w = malloc((numLayers - 1) * sizeof(Matrix *));
	n->b = malloc((numLayers - 1) * sizeof(Matrix *));

	// allocate weight and bias matrices
	for (int i = 0; i < numLayers - 1; i++) {
		n->w[i] = initMatrix(params[i + 1], params[i]);
		n->b[i] = initMatrix(params[i + 1], 1);
	}

	return n;
}

// debug, print matrix
void printMatrix(Matrix * m) {
	for (int r = 0; r < m->rows; r++) {
		for (int c = 0; c < m->cols; c++) {
			printf("%f ", m->at[r][c]);
		}
		printf("\n");
	}
}

void main() {
	int parameters[] = {2, 3, 2};
	NeuralNetwork * n = initNN(3, parameters);
	free(n);
}