import tensorflow as tf
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.1], tf.float32)
linear_model = W * X + b
squared_delta = tf.square(linear_model - Y)
loss = tf.reduce_sum(squared_delta)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
int = tf.global_variables_initializer()

sess = tf.Session();
sess.run(int)
for i in range(1000):
	sess.run(train, {X : [1, 2, 3, 4], Y : [9, 8, 7, 6]})

print (sess.run(loss, {X : [1, 2, 3, 4], Y : [9, 8, 7, 6]}))
print (sess.run([W, b]))

writer = tf.summary.FileWriter('./my_graph', sess.graph)
writer.close()
sess.close()