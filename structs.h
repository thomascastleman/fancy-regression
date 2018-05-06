
#ifndef _STRUCTS_INCLUDED_
#define _STRUCTS_INCLUDED_

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

#endif