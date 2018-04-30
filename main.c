#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "fileio.h"

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

	DataSet * d = readData("/home/tcastleman/Desktop/CS/fancy-regression/MNIST/mnist-train.csv", 10, 0);
	
	return 0;
}