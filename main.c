#include <stdio.h>
#include <stdlib.h>
#include "network.c"

// debug, print matrix
void printMatrix(Matrix * m) {
	for (int r = 0; r < m->rows; r++) {
		for (int c = 0; c < m->cols; c++) {
			printf("%f ", m->at[r][c]);
		}
		printf("\n");
	}
}

void main() {
	// int parameters[] = {2, 3, 2};
	// NeuralNetwork * n = initNN(3, parameters);

	// printMatrix(n->w[0]);
	// free(n);
}