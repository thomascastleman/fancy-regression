
#include "fileio.h"
#include "network.h"

// construct a network off of a serialization
void construct(char * filename, NeuralNetwork * nn);

// write weights and biases of a network to given file
void serialize(char * filename, NeuralNetwork * nn);

// read in and format training data
TrainingData * readTraining() {
	// FILE *fp;
	// char buff[255];

	// fp = fopen("/MNIST/mnist_train.csv", "r");
	// fgets(buff, 255, (FILE*)fp);
	// printf("%s\n", buff);
	// fclose(fp);
}

// turn int into vector form
Matrix * vectorizeIntegerLabel(int label);

// turn character into vector form for comparison with net output
Matrix * vectorizeCharLabel(char label);