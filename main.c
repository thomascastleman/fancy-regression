#include <stdio.h>
#include <stdlib.h>
#include "network.h"
#include "data.h"

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

	int params[] = {2, 3, 1};
	NeuralNetwork * n = initNN(3, params);

	return 0;
}