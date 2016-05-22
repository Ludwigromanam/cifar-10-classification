import tensorflow as tf

from Cifar10DataLoader import loadCifarDataForBatch

cifar10 = loadCifarDataForBatch(1)

print(cifar10)

#Start tensorflow as interactive session
# sess = tf.InteractiveSession()