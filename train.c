#include <stdio.h>
#include <stdlib.h>
#include "structs.h"
#include "meth.h"

// free a double pointer
void freeDP(void ** dp, int size) {
	for (int i = 0; i < size; i++) {
		free(dp[i]);
	}
	free(dp);
}

// set entire network to 0
void zero(NeuralNetwork * n) {
	int l, j, k;
	// weights
	for (l = 0; l < n->numberOfLayers - 1; l++) {
		for (j = 0; j < n->w[l]->rows; j++) {
			for (k = 0; k < n->w[l]->cols; k++) {
				n->w[l]->at[j][k] = 0.0f;
			}
		}
	}

	// biases
	for (l = 0; l < n->numberOfLayers - 1; l++) {
		for (j = 0; j < n->b[l]->rows; j++) {
			n->b[l]->at[j][0] = 0.0f;
		}
	}
}

// swap a training pair within a dataset
void swapPair(DataSet * d, int p1, int p2) {
	Matrix * inp = d->inputs[p1];
	Matrix * out = d->outputs[p1];
	d->inputs[p1] = d->inputs[p2];
	d->outputs[p1] = d->outputs[p2];
	d->inputs[p2] = inp;
	d->outputs[p2] = out;
}

// randomize the order of pairs in a dataset
void shuffle(DataSet * d) {
	// swap each pair with a randomly chosen pair
	for (int n = 0; n < d->size; n++) {
		swapPair(d, n, randInt(0, d->size));
	}
}

// train a given network on a given dataset using batch gradient descent
void train(NeuralNetwork * n, DataSet * training, int batchSize, float learningRate, int epochs) {
	// init net to keep track of gradients for all weights and biases (identical structure)
	NeuralNetwork * gradientNet = initNN(n->numberOfLayers, n->params);

	int L = n->numberOfLayers - 1;	// number of weight matrices / bias vectors

	// allocate arrays of z and delta vectors to hold weighted input and error, respectively
	Matrix ** z = malloc(L * sizeof(Matrix));
	Matrix ** delta = malloc(L * sizeof(Matrix));

	int b, p, l, e;

	// for every epoch
	for (e = 0; e < epochs; e++) {
		printf("Epoch %d\n", e);
		shuffle(training);	// shuffle training data

		// for every batch
		for (b = 0; b < training->size; b += batchSize) {
			zero(gradientNet);	// set entire gradient net to 0

			// for every pair within batch
			for (p = b; p < b + batchSize; p++) {
				// pass input through weights
				z[0] = add(dot(n->w[0], training->inputs[p]), n->b[0]);

				// for every layer
				for (l = 1; l < L; l++) {
					// compute weighted input to lth layer relative to activation in previous
					z[l] = add(dot(n->w[l], sig(z[l - 1])), n->b[l]);
				}

				// calculate last layer error
				delta[L - 1] = hadamard(add(sig(z[L - 1]), scale(-1, training->outputs[p])), sigP(z[L - 1]));

				// record last layer gradients
				gradientNet->w[L - 1] = add(gradientNet->w[L - 1], dot(delta[L - 1], transpose(sig(z[L - 2]))));
				gradientNet->b[L - 1] = add(gradientNet->b[L - 1], delta[L - 1]);

				// moving backward through network
				for (l = L - 2; l >= 0; l--) {
					// calculate error at lth layer
					delta[l] = hadamard(dot(transpose(n->w[l + 1]), delta[l + 1]), sigP(z[l]));

					// add to gradients
					if (l > 0) gradientNet->w[l] = add(gradientNet->w[l], dot(delta[l], transpose(sig(z[l - 1]))));
					gradientNet->b[l] = add(gradientNet->b[l], delta[l]);
				}
				// add first layer weight gradients
				gradientNet->w[0] = add(gradientNet->w[0], dot(delta[0], transpose(sig(training->inputs[p]))));
			}

			// make averaged changes to weights / biases
			for (l = 0; l < L; l++) {
				n->w[l] = add(n->w[l], scale(-learningRate / batchSize, gradientNet->w[l]));
				n->b[l] = add(n->b[l], scale(-learningRate / batchSize, gradientNet->b[l]));
			}
		}
	}

	freeDP((void**) z, L);
	freeDP((void**) delta, L);
}