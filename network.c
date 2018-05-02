#include <stdlib.h>
#include "network.h"

// construct a new r by c matrix
Matrix * initMatrix(int r, int c) {
	Matrix * m = malloc(sizeof(Matrix));
	m->rows = r; m->cols = c;

	// allocate row and column arrays
	m->at = (float **) malloc(r * sizeof(float *));
	for (int i = 0; i < r; i++)
		m->at[i] = (float *) malloc(c * sizeof(float));
	return m;
}

// construct a new network off of given params
NeuralNetwork * initNN(int numLayers, int * params) {
	NeuralNetwork * n = malloc(sizeof(NeuralNetwork));	// allocate network

	n->numberOfLayers = numLayers;
	n->params = params;
	n->w = malloc((numLayers - 1) * sizeof(Matrix *));	// allocate array of weight matrices
	n->b = malloc((numLayers - 1) * sizeof(Matrix *));	// allocate array of bias vectors

	// allocate individual weight and bias matrices
	for (int i = 0; i < numLayers - 1; i++) {
		n->w[i] = initMatrix(params[i + 1], params[i]);
		n->b[i] = initMatrix(params[i + 1], 1);
	}

	return n;
}

// construct new training set
DataSet * initDataSet(int size) {
	DataSet * t = malloc(sizeof(DataSet));
	t->size = size;
	t->inputs = malloc(size * sizeof(Matrix *));
	t->outputs = malloc(size * sizeof(Matrix *));
	return t;
}

// construct a network off of a serialization
void construct(char * filename, NeuralNetwork * nn);

// write weights and biases of a network to given file
void serialize(char * filename, NeuralNetwork * nn);

// train a given network on a given dataset using batch gradient descent
void train(NeuralNetwork * nn, DataSet * training, int batchSize, float learningRate);