# fancy-regression
A feed-forward neural network that learns to recognize handwritten characters using batch gradient descent.

By [Thomas Castleman](https://github.com/thomascastleman) & [Johnny Lindbergh](https://github.com/johnnylindbergh).

Much inspiration was taken from Michael Nielsen's [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/).

### Performance

At one point, we trained a network with 784 input neurons (for processing 28x28 images), 50 hidden neurons, and 10 output neurons on 60,000 training pairs over 10 epochs. Upon testing over 10,000 previously unseen testing pairs, the network achieved 89.1% accuracy. 

Further optimizations could definitely be applied, and this kind of network is by no means state-of-the-art for OCR, but we were satisfied.

### Demo
A version of this network powers the digit recognizing demo [here.](http://castleman.space/ocr/)
