
#ifndef _fileio_
#define _fileio_

void construct(char * filename, NeuralNetwork * nn);

void serialize(char * filename, NeuralNetwork * nn);

TrainingData * readTraining();

#endif