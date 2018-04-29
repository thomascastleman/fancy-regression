#include <stdio.h>
#include <stdlib.h>
#include "fileio.h"
#include "network.h"

// construct a network off of a serialization
void construct(char * filename, NeuralNetwork * nn);

// write weights and biases of a network to given file
void serialize(char * filename, NeuralNetwork * nn);

// read in and format training data
void readData(char * filename) {
	FILE * fp;
	// char buff[1570];

	fp = fopen(filename, "r");

	

	// if (fgets(buff, 1570, fp) != NULL) {
	// 	printf("%s",buff);
	// }
	fclose(fp);
}

// turn int into vector form
Matrix * vectorizeIntegerLabel(int label);

// turn character into vector form for comparison with net output
Matrix * vectorizeCharLabel(char label);