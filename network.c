#include <stdio.h>
#include <stdlib.h>
#include "network.h"

// construct a new r by c matrix
Matrix * initMatrix(int r, int c) {
	Matrix * m = malloc(sizeof(Matrix));
	m->rows = r; m->cols = c;

	// allocate row and column arrays
	m->at = (float **) malloc(r * sizeof(float *));
	for (int i = 0; i < r; i++)
		m->at[i] = (float *) malloc(c * sizeof(float));
	return m;
}

// construct a new network off of given params
NeuralNetwork * initNN(int numLayers, int * params) {
	NeuralNetwork * n = malloc(sizeof(NeuralNetwork));	// allocate network

	n->numberOfLayers = numLayers;
	n->params = params;
	n->w = malloc((numLayers - 1) * sizeof(Matrix *));	// allocate array of weight matrices
	n->b = malloc((numLayers - 1) * sizeof(Matrix *));	// allocate array of bias vectors

	// allocate individual weight and bias matrices
	for (int i = 0; i < numLayers - 1; i++) {
		n->w[i] = initMatrix(params[i + 1], params[i]);
		n->b[i] = initMatrix(params[i + 1], 1);
	}

	return n;
}

// construct new training set
DataSet * initDataSet(int size) {
	DataSet * d = malloc(sizeof(DataSet));
	d->size = size;
	d->inputs = malloc(size * sizeof(Matrix *));
	d->outputs = malloc(size * sizeof(Matrix *));
	return d;
}

// write weights and biases of a network to given file
void serialize(char * filename, NeuralNetwork * n) {
	FILE * fp = fopen(filename, "w");	// open file for writing
	int i, j, k;

	if (fp != NULL) {
		fprintf(fp, "%d|", n->numberOfLayers);

		// write params to file
		for (i = 0; i < n->numberOfLayers - 1; i++) {
			fprintf(fp, "%d,", n->params[i]);
		}
		fprintf(fp, "%d|", n->params[n->numberOfLayers - 1]);

		// for each bias vector
		for (i = 0; i < n->numberOfLayers - 1; i++) {
			// write each bias
			for (j = 0; j < n->params[i + 1]; j++) {
				fprintf(fp, "%f,", n->b[i]->at[j][0]);
			}
		}

		// for each weight matrix
		for (i = 0; i < n->numberOfLayers - 1; i++) {
			// for each row
			for (j = 0; j < n->params[i + 1]; j++) {
				// write each weight
				for (k = 0; k < n->params[i]; k++) {
					fprintf(fp, "%f,", n->w[i]->at[j][k]);
				}
			}
		}
	} else {
		perror("Error opening file (network.c:serialize)");
	}

	fclose(fp);
}

// copy entire file into char * buffer
char * copyFileContents(FILE * fp) {
	// determine file size
	fseek(fp, 0L, SEEK_END);
	long fsize = ftell(fp);
	rewind(fp);

	// allocate buffer
	char * buffer = calloc(1, fsize + 1);
	if (!buffer) fclose(fp), exit(1);

	// copy file contents into buffer
	if (fread(buffer, fsize, 1, fp) != 1)
		free(buffer), exit(1);
	fclose(fp);

	return buffer;
}

// copy everything in buffer up to stopping point into a char *
char * readNextValue(char * buffer, int * buffIndex, char stop) {
	int size = 0, i = *buffIndex, k = 0;
	while (buffer[i++] != stop)
		size++;

	// fill "next" with value up to stop character
	char * next = malloc(size * sizeof(char));

	printf("\nCall:\n");

	// copy selected portion into next, moving along buffer index
	i--;
	while (*buffIndex < i) {
		printf("Character: %c\n", buffer[*buffIndex]);
		next[k++] = buffer[(*buffIndex)++];
	}
	(*buffIndex)++;

	return next;
}

// construct a network off of a serialization
NeuralNetwork * construct(char * filename) {
	FILE * fp = fopen(filename, "r");	// open file for reading
	int i = 0, l, j, k;

	if (fp != NULL) {
		char * buffer = copyFileContents(fp);	// copy contents into buffer for reading

		// read number of layers and allocate corresponding parameters array
		int numLayers = atoi(readNextValue(buffer, &i, '|'));
		int params[numLayers], pInd = 0;
		char stop = ',';

		// for each parameter
		while (pInd < numLayers) {
			if (pInd == numLayers - 1)
				stop = '|';	// change delimiter if last param
			params[pInd++] = atoi(readNextValue(buffer, &i, stop));
		}

		// initialize network
		NeuralNetwork * n = initNN(numLayers, params);

		// for every bias vector
		for (l = 0; l < numLayers - 1; l++) {
			// for each bias
			for (j = 0; j < params[l + 1]; j++) {
				n->b[l]->at[j][0] = atof(readNextValue(buffer, &i, ','));
			}
		}

		// for every weight matrix
		for (l = 0; l < numLayers - 1; l++) {
			// for every row
			for (j = 0; j < params[l + 1]; j++) {
				// for every column
				for (k = 0; k < params[l]; k++) {
					n->w[l]->at[j][k] = atof(readNextValue(buffer, &i, ','));
				}
			}
		}

		free(buffer);
		return n;
	} else {
		perror("Error reading file (network.c:construct)"), exit(1);
	}
}