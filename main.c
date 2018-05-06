#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "structs.h"
#include "data.h"
#include "train.h"
#include "meth.h"

// randomize weights and biases within ranges
void randomizeNet(NeuralNetwork * n, float wMin, float wMax, float bMin, float bMax) {
	for (int l = 0; l < n->numberOfLayers - 1; l++) {
		randomize(n->w[l], wMin, wMax);
		randomize(n->b[l], bMin, bMax);
	}
}

int main() {
	srand(time(NULL));

	int params[] = {784, 10, 10, 10, 36};
	NeuralNetwork * n = initNN(5, params);
	randomizeNet(n, -0.25, 0.25, -1, 1);

	DataSet * mnist = readMNIST("/home/tcastleman/Desktop/CS/fancy-regression/MNIST/mnist-train.csv", 1, 0);

	train(n, mnist, 1, 1);

	return 0;
}