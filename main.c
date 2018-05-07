#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "structs.h"
#include "data.h"
#include "train.h"
#include "meth.h"
#include "storage.h"

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

int main() {
	srand(time(NULL));

	int params[] = {784, 50, 36};
	NeuralNetwork * n = initNN(3, params);
	randomizeNet(n, -0.5, 0.5, -0.5, 0.5);

	DataSet * mnist = readMNIST("/home/tcastleman/Desktop/CS/fancy-regression/MNIST/mnist-train.csv", 2500, 0);
	train(n, mnist, 10, 1, 10);

	// serialize("/home/tcastleman/Desktop/CS/fancy-regression/net.txt", n);

	Matrix * output = forwardPass(n, mnist->inputs[0]);
	printf("\nOUTPUT!!!\n");
	printMatrix(output);

	printf("Actual:\n");
	printMatrix(mnist->outputs[0]);

	return 0;
}