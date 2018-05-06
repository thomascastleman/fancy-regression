
#ifndef _STORAGE_INCLUDED_
#define _STORAGE_INCLUDED_

void serialize(char * filename, NeuralNetwork * n);

NeuralNetwork * construct(char * filename);

#endif