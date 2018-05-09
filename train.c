#include <stdio.h>
#include <stdlib.h>
#include "structs.h"
#include "meth.h"
#include "util.h"

#include <math.h>

// compute weighted input in layer l
Matrix * weightedInput(int l, Matrix * w_l, Matrix * b_l, Matrix * a_prev) {
	Matrix * weighted = dot(w_l, a_prev);
	Matrix * plusBias = add(weighted, b_l);
	freeMatrix(weighted);
	return plusBias;
}

// compute error in last layer of network relative to weighted input and expected output
Matrix * lastError(int l, Matrix * z, Matrix * y) {
	Matrix * a_L = sig(z);
	Matrix * neg_y = scale(-1, y);
	Matrix * diff = add(a_L, neg_y);
	Matrix * sigPrimeZ = sigP(z);

	Matrix * err = hadamard(diff, sigPrimeZ);

	freeMatrix(a_L);
	freeMatrix(neg_y);
	freeMatrix(diff);
	freeMatrix(sigPrimeZ);

	return err;
}

// compute error in lth layer in terms of error in (l+1)th layer
Matrix * error(int l, Matrix * z, Matrix * w_l_next, Matrix * d_next) {
	Matrix * weightT = transpose(w_l_next);
	Matrix * w_dot_delta = dot(weightT, d_next);
	Matrix * sigPrimeZ = sigP(z);

	Matrix * err = hadamard(w_dot_delta, sigPrimeZ);

	freeMatrix(weightT);
	freeMatrix(w_dot_delta);
	freeMatrix(sigPrimeZ);	

	return err;
}

// calculate a weight gradient given delta and previous activation
Matrix * weightGradient(Matrix * d, Matrix * a_prev) {
	Matrix * a_trans = transpose(a_prev);
	Matrix * d_dot_a_trans = dot(d, a_trans);
	freeMatrix(a_trans);
	return d_dot_a_trans;
}

// train a given network on a given dataset using batch gradient descent
void train(NeuralNetwork * n, DataSet * training, int batchSize, float learningRate, int epochs) {
	// init net to keep track of gradients for all weights and biases (identical structure)
	NeuralNetwork * gradientNet = initNN(n->numberOfLayers, paramCopy(n->params, n->numberOfLayers));

	int L = n->numberOfLayers - 1;	// number of weight matrices / bias vectors

	// allocate arrays of z and delta vectors to hold weighted input and error, respectively
	Matrix ** z = malloc(L * sizeof(Matrix));
	Matrix ** delta = malloc(L * sizeof(Matrix));

	int b, p, l, e;	// iterating variables
	Matrix *prevAct, *oldBiases, *oldWeights, *gradient;	// temp matrix pointers

	// for every epoch
	for (e = 0; e < epochs; e++) {
		printf("Epoch %d\n", e);
		shuffle(training);	// shuffle training data

		// for every batch
		for (b = 0; b < training->size; b += batchSize) {

			// for every pair within batch
			for (p = b; p < b + batchSize; p++) {

				// for every layer, propagate forward
				for (l = 0; l < L; l++) {
					// compute weighted input to lth layer relative to activation in previous
					prevAct = l == 0 ? training->inputs[p] : sig(z[l - 1]);
					z[l] = weightedInput(l, n->w[l], n->b[l], prevAct);
					if (l > 0) freeMatrix(prevAct);
				}

				// moving backward through network (they call it BACKpropagate for a reason)
				for (l = L - 1; l >= 0; l--) {
					if (l ==  L - 1) {
						// compute error at last layer
						delta[l] = lastError(l, z[l], training->outputs[p]);
					} else {
						// recursively calculate error at lth layer
						delta[l] = error(l, z[l], n->w[l + 1], delta[l + 1]);

						// next layer error no longer needed
						freeMatrix(delta[l + 1]);
					}

					// add to sum of bias gradients
					oldBiases = gradientNet->b[l];
					gradientNet->b[l] = add(gradientNet->b[l], delta[l]);
					freeMatrix(oldBiases);

					// add to sum of weight gradients
					prevAct = l == 0 ? training->inputs[p] : sig(z[l - 1]);
					oldWeights = gradientNet->w[l];
					gradient = weightGradient(delta[l], prevAct);
					gradientNet->w[l] = add(gradientNet->w[l], gradient);
					freeMatrix(gradient);
					freeMatrix(oldWeights);
					if (l > 0) freeMatrix(prevAct);

					freeMatrix(z[l]); // this layer z no longer needed
				}

				freeMatrix(delta[0]);	// free first layer error
			}

			// make averaged changes to weights / biases
			for (l = 0; l < L; l++) {
				gradient = scale(-learningRate / batchSize, gradientNet->w[l]);
				oldWeights = n->w[l];
				n->w[l] = add(n->w[l], gradient);
				freeMatrix(oldWeights);
				freeMatrix(gradient);

				gradient = scale(-learningRate / batchSize, gradientNet->b[l]);
				oldBiases = n->b[l];
				n->b[l] = add(n->b[l], gradient);
				freeMatrix(oldBiases);
				freeMatrix(gradient);

				// reset gradient net values
				zero(gradientNet->w[l]);
				zero(gradientNet->b[l]);
			}
		}
	}

	free(z);
	free(delta);
	freeNetwork(gradientNet);
}