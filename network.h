
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

void construct(char * filename, NeuralNetwork * nn);

void serialize(char * filename, NeuralNetwork * nn);

void train(NeuralNetwork * nn, DataSet * training, int batchSize, float learningRate);

#endif