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