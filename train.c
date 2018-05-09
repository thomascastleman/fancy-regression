#include <stdio.h>
#include <stdlib.h>
#include "structs.h"
#include "meth.h"
#include "util.h"

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
	NeuralNetwork * gradientNet = initNN(n->numberOfLayers, paramCopy(n->params, n->numberOfLayers));

	int L = n->numberOfLayers - 1;	// number of weight matrices / bias vectors

	// allocate arrays of z and delta vectors to hold weighted input and error, respectively
	Matrix ** z = malloc(L * sizeof(Matrix));
	Matrix ** delta = malloc(L * sizeof(Matrix));

	int b, p, l, e;

	Matrix *s, *prod, *scaled, *sum, *sP, *tr, *weights, *biases;

	// for every epoch
	for (e = 0; e < epochs; e++) {
		printf("Epoch %d\n", e);
		shuffle(training);	// shuffle training data

		// for every batch
		for (b = 0; b < training->size; b += batchSize) {
			zero(gradientNet);	// set entire gradient net to 0

			// for every pair within batch
			for (p = b; p < b + batchSize; p++) {

				if (e > 0) freeMatrix(z[0]);

				// pass input through weights
				prod = dot(n->w[0], training->inputs[p]);
				z[0] = add(prod, n->b[0]);
				freeMatrix(prod);


				// for every layer
				for (l = 1; l < L; l++) {
					// compute weighted input to lth layer relative to activation in previous
					s = sig(z[l - 1]);
					prod = dot(n->w[l], s);

					if (e > 0) freeMatrix(z[l]);
					z[l] = add(prod, n->b[l]);

					freeMatrix(s);
					freeMatrix(prod);
				}

				s = sig(z[L - 1]);
				scaled = scale(-1, training->outputs[p]);
				sum = add(s, scaled);
				sP = sigP(z[L - 1]);

				// calculate last layer error
				if (e > 0) freeMatrix(delta[L - 1]);
				delta[L - 1] = hadamard(sum, sP);

				freeMatrix(s);
				freeMatrix(scaled);
				freeMatrix(sum);
				freeMatrix(sP);

				s = sig(z[L - 2]);
				tr = transpose(s);
				prod = dot(delta[L - 1], tr);

				weights = gradientNet->w[L - 1];
				biases = gradientNet->b[L - 1];

				// record last layer gradients
				gradientNet->w[L - 1] = add(gradientNet->w[L - 1], prod);
				gradientNet->b[L - 1] = add(gradientNet->b[L - 1], delta[L - 1]);

				freeMatrix(weights);
				freeMatrix(biases);

				freeMatrix(s);
				freeMatrix(tr);
				freeMatrix(prod);

				// moving backward through network
				for (l = L - 2; l >= 0; l--) {
					tr = transpose(n->w[l + 1]);
					prod = dot(tr, delta[l + 1]);
					sP = sigP(z[l]);

					if (e > 0) freeMatrix(delta[l]);
					// calculate error at lth layer
					delta[l] = hadamard(prod, sP);

					freeMatrix(tr);
					freeMatrix(prod);
					freeMatrix(sP);

					biases = gradientNet->b[l];
					// add to gradients
					gradientNet->b[l] = add(gradientNet->b[l], delta[l]);
					freeMatrix(biases);

					if (l > 0) {
						s = sig(z[l - 1]);
						tr = transpose(s);
						prod = dot(delta[l], tr);

						weights = gradientNet->w[l];
						gradientNet->w[l] = add(gradientNet->w[l], prod);
						freeMatrix(weights);
	
						freeMatrix(s);
						freeMatrix(tr);
						freeMatrix(prod);

					}
				}
				s = sig(training->inputs[p]);
				tr = transpose(s);
				prod = dot(delta[0], tr);

				weights = gradientNet->w[0];
				// add first layer weight gradients
				gradientNet->w[0] = add(gradientNet->w[0], prod);
				freeMatrix(weights);

				freeMatrix(s), freeMatrix(tr), freeMatrix(prod);
			}

			// make averaged changes to weights / biases
			for (l = 0; l < L; l++) {
				scaled = scale(-learningRate / batchSize, gradientNet->w[l]);
				weights = n->w[l];
				n->w[l] = add(n->w[l], scaled);
				freeMatrix(weights);
				freeMatrix(scaled);

				scaled = scale(-learningRate / batchSize, gradientNet->b[l]);
				biases = n->b[l];
				n->b[l] = add(n->b[l], scaled);
				freeMatrix(biases);
				freeMatrix(scaled);
			}
		}
	}

	for (e = 0; e < L; e++) {
		freeMatrix(z[e]);
		freeMatrix(delta[e]);
	}

	freeNetwork(gradientNet);
}