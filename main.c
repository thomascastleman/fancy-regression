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
}

int main() {
	srand(time(NULL));
	Matrix * studentA = initMatrix(5,6);
	Matrix * studentB = initMatrix(5,5);
	randomize(studentA, -1,1);
	randomize(studentB,-1,1);
	printMatrix(studentA);
	printMatrix(studentB);
	Matrix * disciplinaryCase = dot(studentA,studentB);
	return 0;
}