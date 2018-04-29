#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fileio.h"
#include "network.h"

// construct a network off of a serialization
void construct(char * filename, NeuralNetwork * nn);

// write weights and biases of a network to given file
void serialize(char * filename, NeuralNetwork * nn);

// read in and format training data
DataSet * readData(char * filename, int size) {
	FILE * fp;
	// char buff[1570];

	fp = fopen(filename, "r");

	// if (fgets(buff, 1570, fp) != NULL) {
	// 	printf("%s",buff);
	// }

	DataSet * d = initDataSet(size);
	int c, i = 0;
	int label, comma;

	int rowIndex = 0;	// current index in current vector


	if (fp != NULL) {





		// while (i++ < 50) {
		//     label = fgetc(fp);
		//     comma = fgetc(fp);

		//     if(feof(fp)) {
		//     	break;
		//     }

		//     if (c == ',') {
		//     	// printf("It's a comma");
		//     }
		//     if (c == '\n') {
		    	
		//     }

		//     printf("%c\n", c);
		// }

	}
	
	fclose(fp);


	return d;
}

int charStarToInt(char * x);

// turn int into vector form
Matrix * vectorizeIntegerLabel(int label);

// turn character into vector form for comparison with net output
Matrix * vectorizeCharLabel(char label);