import tensorflow as tf

#Load data from CIFAR-10 dataset
#yields a dictionary where "data" contains the data, and "label" contains the true classification
def unpickleBatch(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def fileForBatch(batchNum):
	return "cifar-10-batches-py/data_batch_%d"%(batchNum)

cifar10 = unpickleBatch(fileForBatch(1))

print(cifar10)

#Start tensorflow as interactive session
# sess = tf.InteractiveSession()