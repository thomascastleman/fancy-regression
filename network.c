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

typedef struct {
	int size;
	char * value;
} String;

// copy everything in buffer up to stopping point into a char *
String * readNextValue(char * buffer, int position, char stop) {
	int size = 0, i = position, k = 0;
	while (buffer[i++] != stop)
		size++;

	String * s = malloc(sizeof(String));
	s->size = size;
	s->value = malloc(size * sizeof(char));	// allocate enough to store this value

	for (i = position; i < position + size; i++) {
		s->value[k++] = buffer[i];
	}

	return s;
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

	return buffer;
}

// construct a network off of a serialization
NeuralNetwork * construct(char * filename) {
	FILE * fp = fopen(filename, "r");	// open file for reading
	int i, l, j, k;
	String * next;

	if (fp != NULL) {
		char * buffer = copyFileContents(fp);	// copy contents into buffer for reading
		fclose(fp);

		next = readNextValue(buffer, 0, '|');	// parse number of layers
		int numLayers = atoi(next->value);
		i = next->size + 1;	// move index along in buffer

		int params[numLayers], paramIndex = 0;	// allocate parameters array
		char stop = ',';

		// for each parameter
		while (paramIndex < numLayers) {
			if (paramIndex == numLayers - 1) stop = '|';	// change delimiter if last param
			next = readNextValue(buffer, i, stop);		// read parameter into param array
			params[paramIndex++] = atoi(next->value);
			i += next->size + 1;	// move index along
		}

		// initialize network
		NeuralNetwork * n = initNN(numLayers, params);

		// for every bias vector
		for (l = 0; l < numLayers - 1; l++) {
			// for each bias
			for (j = 0; j < params[l + 1]; j++) {
				next = readNextValue(buffer, i, ',');
				n->b[l]->at[j][0] = atof(next->value);
				i += next->size + 1;
			}
		}

		// for each weight matrix
		for (l = 0; l < numLayers - 1; l++) {
			for (j = 0; j < params[l + 1]; j++) {
				for (k = 0; k < params[l]; k++) {
					next = readNextValue(buffer, i, ',');
					n->w[l]->at[j][k] = atof(next->value);
					i += next->size + 1;
				}
			}
		}

		free(buffer);
		return n;
	} else {
		perror("Error reading file (network.c:construct)"), exit(1);
	}
}