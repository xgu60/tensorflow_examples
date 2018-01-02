from examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 
import matplotlib.pyplot as plt

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1,], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
		padding='SAME')

#input data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#construct a cnn model
#first convolutional layer
#the convolution will compute 32 features for each 5 x 5 patch
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second convolutional layer
#the convolution will compute 64 features for each 5 x 5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_predict = tf.nn.softmax(y_conv)




#cross-entropy function
#cross_entropy =  tf.reduce_mean(-tf.reduce_mean(y * tf.log(y_predict), 
#	reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	labels=y, logits=y_predict))

#optimizer
optimizer = tf.train.AdamOptimizer(1e-4)

#trainer
train =  optimizer.minimize(cross_entropy)

#initilizer
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#model evaluation
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#start training
train_accuracy = []
#test_accuracy = []
for _ in range(10000):
	batch_xs, batch_xy = mnist.train.next_batch(50)
	sess.run(train, feed_dict={x: batch_xs, y: batch_xy, keep_prob: 1.0})
	train_accuracy.append(sess.run(accuracy, 
		feed_dict={x: batch_xs, y: batch_xy, keep_prob: 1.0}))
	#test_accuracy.append(sess.run(accuracy, 
	#	feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))


#plot the accuracy of training batch
#plt.plot(list(range(100)), train_accuracy, 'r', list(range(100)), test_accuracy, 'b')
plt.plot(list(range(10000)), train_accuracy, 'r')
plt.ylabel('accuracy for training batches')
plt.show()

#print(sess.run(accuracy, 
#	feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
#print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_xy}))


