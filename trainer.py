import tensorflow as tf
from Cifar10DataLoader import loadCifarDataForBatch, loadTestBatch

#TODO: this should be set with a command line parmaeter
LOGGING_DIR = "logging"

#Start tensorflow as interactive session
sess = tf.InteractiveSession()


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#Log information about the given tensor into a histogram
def writeHistogramSummary(label, tensor):
  with tf.name_scope(label):
    print "histogram ", label, " shape:", tensor.get_shape()
    tf.scalar_summary("%s max: " % label, tf.reduce_max(tensor))
    tf.scalar_summary("%s min: " % label, tf.reduce_min(tensor))
    tf.scalar_summary("%s mean: " % label, tf.reduce_mean(tensor))
    tf.histogram_summary(label, tensor)

#Log information about the given tensor as a scalar
def writeScalarSummary(label, tensor):
  with tf.name_scope(label):
    print "scalar ", label, " shape:", tensor.get_shape()
    tf.scalar_summary(label, tensor)


x = tf.placeholder(tf.float32, [None, 3072]) 
y_ = tf.placeholder(tf.float32, [None, 10])

#First Convolutional Layer - return 32 features by sampling 5x5 areas, over 3 color channels
with tf.name_scope("first_layer"):  
  W_conv1 = weight_variable([5,5,3,32])
  # writeHistogramSummary("weight", W_conv1)

  b_conv1 = bias_variable([32])
  # writeHistogramSummary("bias", b_conv1)

  x_image = tf.reshape(x, [-1,32,32,3])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  writeHistogramSummary("first_layer/activations", h_conv1)

  h_pool1 = max_pool_2x2(h_conv1)
  writeHistogramSummary("pooling", h_pool1)



#Second Convolutional Layer - return 64 features by sampling 5x5 areas
with tf.name_scope("second_layer"):
  W_conv2 = weight_variable([5,5,32,64])
  # writeHistogramSummary("weight", W_conv2)

  b_conv2 = bias_variable([64])
  # writeHistogramSummary("bias", b_conv2)


  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  writeHistogramSummary("second_layer/activations", h_conv2)

  h_pool2 = max_pool_2x2(h_conv2)
  # writeHistogramSummary("pooling", h_pool2)



#Deeply Connected Layer - TODO why is this value not 8*8*64*3?
with tf.name_scope("deeply_connected_layer"):
  W_fc1 = weight_variable([8*8*64, 1024])
  # writeHistogramSummary("weight", W_fc1)

  b_fc1 = bias_variable([1024])
  # writeHistogramSummary("bias", b_fc1)


  h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  writeHistogramSummary("deeply_connected_layer/activations", h_fc1)


#Dropout
with tf.name_scope("dropout_layer"):
  keep_prob = tf.placeholder(tf.float32)
  writeHistogramSummary("dropout_layer/keep_prob", keep_prob)

  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  writeHistogramSummary("dropout_layer/dropout", h_fc1_drop)


#Readout Layer
with tf.name_scope("readout_layer"):
  W_fc2 = weight_variable([1024, 10])
  # writeHistogramSummary("weight", W_fc2,)

  b_fc2 = bias_variable([10])
  # writeHistogramSummary("bias", b_fc2)

  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # writeScalarSummary("softmax_result_y", y_conv)


#Train and Evaluate the Model

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
writeScalarSummary("cross_entropy", cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
writeScalarSummary("accuracy", accuracy)  


#Aggregate logging
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(LOGGING_DIR + "/train", sess.graph)
test_writer = tf.train.SummaryWriter(LOGGING_DIR + "/test")

#run the session
sess.run(tf.initialize_all_variables())

for b in range(1,6):
  batch = loadCifarDataForBatch(b, oneHot = True)
  data_for_batch = batch["data"]
  labels_for_batch = batch["labels"]
  for i in range(0,10000,50):
    # print("starting run for %s"%(i))
    #use 50 values per step
    x_input = data_for_batch[i:i+50,:]
    y_input = labels_for_batch[i:i+50,:]

    # print(x_input.shape)
    # print(y_input.shape)

    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict = {
        x: x_input, y_: y_input, keep_prob: 1.0 })
      print("step %d, training accuracy %g" % (i, train_accuracy))

    #Run a train step every iteration
    #(also run the merged function which will aggregate summary data for logging)
    # train_step.run(feed_dict={x: x_input, y_: y_input, keep_prob: .5})
    summary,acc = sess.run([merged, train_step], feed_dict={x: x_input, y_: y_input, keep_prob: .5})
    test_writer.add_summary(summary, i)

  print("done with batch %d" % b)


test_batch = loadTestBatch(oneHot = True)

print ("test accuracy: %g" % accuracy.eval(feed_dict={
  x: test_batch["data"], y_: test_batch["labels"], keep_prob: 1.0}))
