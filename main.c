#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "structs.h"
#include "data.h"
#include "train.h"
#include "meth.h"
#include "storage.h"
#include "util.h"

int main() {
	srand(time(NULL));	// set random seed

	// // TRAINING --------------------------------

	// int params[] = {784, 50, 36};
	// NeuralNetwork * n = initNN(sizeof(params) / sizeof(int), params);
	// gaussianRandomizeNet(n);

	// DataSet * mnist = readMNIST("/home/tcastleman/Desktop/CS/fancy-regression/MNIST/mnist-train.csv", 2000, 0);
	// train(n, mnist, 10, 0.05, 10);
	// serialize("/home/tcastleman/Desktop/CS/fancy-regression/net.txt", n);



	// // TESTING -----------------------------------

	// NeuralNetwork * n = construct("/home/tcastleman/Desktop/CS/fancy-regression/net.txt");
	// DataSet * test = readMNIST("/home/tcastleman/Desktop/CS/fancy-regression/MNIST/mnist-test.csv", 10000, 0);
	// printf("Accuracy: %f\n", accuracy(n, test));

	// Matrix * est = forwardPass(n, test->inputs[0]);
	// printSideBySide(est, test->outputs[0]);

	freeNetwork(n);
	return 0;
}