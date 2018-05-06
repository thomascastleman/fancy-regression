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
	Matrix * studentA = initMatrix(2,2);
	Matrix * studentB = initMatrix(2,3);
	randomize(studentA, -1,1);
	randomize(studentB,-1,1);
	printf("A\n");
	printMatrix(studentA);
	printf("B\n");
	printMatrix(studentB);
	Matrix * disciplinaryCase = dot(studentA,studentB);
	printf("C\n");
	printMatrix(disciplinaryCase);
	return 0;
}