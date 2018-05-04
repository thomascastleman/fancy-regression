#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"
#include "data.h"
#include "meth.h"

// debug, print matrix
void printMatrix(Matrix * m) {
	for (int r = 0; r < m->rows; r++) {
		for (int c = 0; c < m->cols; c++) {
			printf("%f ", m->at[r][c]);
		}
		printf("\n");
	}
}

int main() {
	// srand(time(NULL));




	int params[] = {20, 10, 5};
	NeuralNetwork * n = initNN(3, params);
	serialize("/home/tcastleman/Desktop/CS/fancy-regression/test.txt", n);
	printf("Finished serializing\n");
	NeuralNetwork * copy = construct("/home/tcastleman/Desktop/CS/fancy-regression/test.txt");
	printf("Finished constructing\n");
	free(n);

	return 0;
}