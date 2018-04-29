
#ifndef _network_
#define _network_

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
} TrainingData;

Matrix * initMatrix(int r, int c);

NeuralNetwork * initNN(int numLayers, int * params);

TrainingData * initTrainingData(int size);

void train(NeuralNetwork * nn, TrainingData * training, int batchSize, float learningRate);

#endif