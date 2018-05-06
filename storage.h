
#ifndef _STORAGE_INCLUDED_
#define _STORAGE_INCLUDED_

#include "structs.h"

void serialize(char * filename, NeuralNetwork * n);

NeuralNetwork * construct(char * filename);

#endif