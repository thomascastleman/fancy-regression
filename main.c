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
	srand(time(NULL));

	int params[] = {784, 50, 36};
	NeuralNetwork * n = initNN(3, params);
	randomizeNet(n, -0.5, 0.5, -0.5, 0.5);

	DataSet * mnist = readMNIST("/home/tcastleman/Desktop/CS/fancy-regression/MNIST/mnist-train.csv", 30000, 0);
	
	train(n, mnist, 20, 1, 3);

	// serialize("/home/tcastleman/Desktop/CS/fancy-regression/net.txt", n);

	// NeuralNetwork * n = construct("/home/tcastleman/Desktop/CS/fancy-regression/net.txt");
	// DataSet * test = readMNIST("/home/tcastleman/Desktop/CS/fancy-regression/MNIST/mnist-test.csv", 1000, 0);
	// printf("Accuracy: %f\n", accuracy(n, test));

	return 0;
}