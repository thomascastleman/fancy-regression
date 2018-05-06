#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "structs.h"
#include "data.h"
#include "train.h"
#include "meth.h"

// debug, print matrix
void printMatrix(Matrix * m) {
	for (int r = 0; r < m->rows; r++) {
		for (int c = 0; c < m->cols; c++) {
			printf("%f ", m->at[r][c]);
		}
		printf("\n");
	}
	printf("\n");
}

// randomize weights and biases within ranges
void randomizeNet(NeuralNetwork * n, float wMin, float wMax, float bMin, float bMax) {
	for (int l = 0; l < n->numberOfLayers - 1; l++) {
		randomize(n->w[l], wMin, wMax);
		randomize(n->b[l], bMin, bMax);
	}
}

// pass an input vector through a network
Matrix * forwardPass(NeuralNetwork * n, Matrix * input) {
	Matrix * act = input;

	// pass activation through network
	for (int l = 0; l < n->numberOfLayers - 1; l++) {
		act = sig(add(dot(n->w[l], act), n->b[l]));
	}

	return act;
}

int main() {
	srand(time(NULL));

	int params[] = {784, 10, 10, 10, 36};
	NeuralNetwork * n = initNN(5, params);
	randomizeNet(n, -0.25, 0.25, -1, 1);

	DataSet * mnist = readMNIST("/Users/johnlindbergh/Documents/fancy-regression/MNIST/mnist-train.csv", 10000, 0);

	train(n, mnist, 5, 1);

	Matrix * output = forwardPass(n, mnist->inputs[0]);
	printf("OUTPUT!!!\n");
	printMatrix(output);

	printf("Actual:\n");
	printMatrix(mnist->outputs[0]);

	return 0;
}