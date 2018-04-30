#include <stdio.h>
#include <stdlib.h>
#include "fileio.h"
#include "network.h"

#define SIZE_OF_IMAGE 784 // 28x28 training images
#define OUTPUT_VECTOR_SIZE 36	// 10 digits, 26 letters

// construct a network off of a serialization
void construct(char * filename, NeuralNetwork * nn);

// write weights and biases of a network to given file
void serialize(char * filename, NeuralNetwork * nn);

// take an int array representing the place values of an integer and convert to actual value
int convertToInt(int places[], int i) {
	int value = 0, placeValue = 1;
	while (i >= 0) {
		value += placeValue * places[i--];
		placeValue *= 10;
	}
	return value;
}

// create output vector out of label (first digits 0-9, then letters 1-26);
Matrix * vectorizeLabel(int value, int isLetter) {
	Matrix * v = initMatrix(OUTPUT_VECTOR_SIZE, 1);
	if (!isLetter) {
		v->at[value][0] = 1.0;
	} else {
		v->at[value + 9][0] = 1.0;
	}
	return v;
}

// construct a data set of image vector inputs and label vector ouputs off of a csv file
DataSet * readData(char * filename, int size, int usingLetters) {
	DataSet * d = initDataSet(size);

	FILE * fp;	// declare pointer to filestream
	fp = fopen(filename, "r");	// open file for reading

	if (fp != NULL) {
		int c, i = 0, pairIndex = 0, label[2], pixel[3];

		while (pairIndex < size) {
			// establish label as output vector
			while ((c = fgetc(fp)) != ',') {
				label[i++] = c - '0';
			}
			int labelValue = convertToInt(label, i - 1);
			d->outputs[pairIndex] = vectorizeLabel(labelValue, usingLetters);

			i = 0;
			int vectorIndex = 0;
			Matrix * inputVector = initMatrix(SIZE_OF_IMAGE, 1);

			// while end of image not yet reached
			while ((c = fgetc(fp)) != '\n') {
				// if end of pixel
				if (c == ',') {
					inputVector->at[vectorIndex++][0] = convertToInt(pixel, i - 1);	// add pixel intensity to input vector
					i = 0;
				} else {
					pixel[i++] = c - '0';	// get int representation of next char
				}
			}

			// add input vector to dataset
			d->inputs[pairIndex++] = inputVector;

			// if at end, break
			if (feof(fp))
				break;
		}
	} else {
		perror("Error reading file (fileio.c)");
	}

	fclose(fp);
	return d;
}