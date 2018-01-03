import os
from six.moves.urllib.request import urlopen

import numpy as np 
import tensorflow as tf 

#data sets
iris_training = "iris_training.csv"
iris_training_url = "http://download.tensorflow.org/data/iris_training.csv"

iris_test = "iris_test.csv"
iris_test_url = "http://download.tensorflow.org/data/iris_test.csv"

def main():
	#download dataset if it is not stored locally
	if not os.path.exists(iris_training):
		raw = urlopen(iris_training_url).read()
		with open(iris_training, "wb") as f:
			f.write(raw)

	if not os.path.exists(iris_test):
		raw = urlopen(iris_test_url).read()
		with open(iris_test, "wb") as f:
			f.write(raw)

	#load dataset
	training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
		filename=iris_training,
		target_dtype=np.int,
		features_dtype=np.float32)
	test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
		filename=iris_test,
		target_dtype=np.int,
		features_dtype=np.float32)

	#specify that all features have real-value data
	feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

	#build a three layer DNN with 10, 20, 10 units respectively.
	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
											hidden_units=[10, 20, 10],
											n_classes=3,
											model_dir="/tmp/iris_model")
	#define the training inputs
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(training_set.data)},
		y=np.array(training_set.target),
		num_epochs=None,
		shuffle=True)

	#train model
	classifier.train(input_fn=train_input_fn, steps=2000)

	#define test input
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(test_set.data)},
		y=np.array(test_set.target),
		num_epochs=1,
		shuffle=False)

	#evaluate accuracy
	accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))












if __name__ == "__main__":
	main()
