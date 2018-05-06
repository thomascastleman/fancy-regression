#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "structs.h"
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

int main() {
	srand(time(NULL));

	return 0;
}