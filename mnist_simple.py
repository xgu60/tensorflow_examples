from examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 
import matplotlib.pyplot as plt
#input data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implement model
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)

#cross-entropy function
#cross_entropy =  tf.reduce_mean(-tf.reduce_mean(y * tf.log(y_predict), 
#	reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)

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
test_accuracy = []
for _ in range(1000):
	batch_xs, batch_xy = mnist.train.next_batch(100)
	sess.run(train, feed_dict={x: batch_xs, y: batch_xy})
	train_accuracy.append(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_xy}))
	test_accuracy.append(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


#plot the accuracy of training batch
plt.plot(list(range(1000)), train_accuracy, 'r', list(range(1000)), test_accuracy, 'b')
plt.ylabel('accuracy for training batches')
plt.show()

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
#print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_xy}))


