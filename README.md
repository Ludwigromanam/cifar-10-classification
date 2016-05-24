# CIFAR-10 Convolutional Neural Network

A basic convolutional neural network trainer to classify images from the CIFAR-10 image dataset implemented with [TensorFlow](https://www.tensorflow.org/). 
The CIFAR-10 dataset is a collection of colored images to a set of classifications of what the image depicts. More 
information about the CIFAR-10 dataset can be found [here](http://www.cs.utoronto.ca/~kriz/cifar.html).


Note: This is a practice project for me to learn and get familiar with convolutional neural networks. I will be using it to play around with tweaking hyperparameters and the network structure, so the learning rates will probably be quite low for a while.


##Usage

Simply run
```
python trainer.py
```

If the CIFAR-10 dataset is not found, it will automatically be downloaded to `cifar-10-batches-py/`.



##Reference

http://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf
