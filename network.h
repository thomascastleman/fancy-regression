
#ifndef _NETWORK_INCLUDED_
#define _NETWORK_INCLUDED_

typedef struct {
	int rows;
	int cols;
	float ** at;
} Matrix;

typedef struct {
	int numberOfLayers;
	int * params;
	Matrix ** w;
	Matrix ** b;
} NeuralNetwork;

typedef struct {
	int size;
	Matrix ** inputs;
	Matrix ** outputs;
} DataSet;

Matrix * initMatrix(int r, int c);

NeuralNetwork * initNN(int numLayers, int * params);

DataSet * initDataSet(int size);

void serialize(char * filename, NeuralNetwork * n);

void construct(char * filename, NeuralNetwork * n);

void train(NeuralNetwork * n, DataSet * training, int batchSize, float learningRate);

#endif