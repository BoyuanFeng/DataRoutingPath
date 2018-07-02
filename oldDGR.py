# In this file, the dataset contains the coarse label 0,1,2,3,14 in CIFAR100 according to b"coarse_label"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)
from scipy.stats.stats import pearsonr

import keras
#from keras.datasets import cifar100
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# shuffle function is here, used for traning time
def shuffle_images_and_labels(images, labels):
  rand_indexes = np.random.permutation(images.shape[0])
  shuffled_images = images[rand_indexes]
  shuffled_labels = labels[rand_indexes]
  return shuffled_images, shuffled_labels

def unpickle(file):
  import pickle
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding = 'bytes')
  fo.close()
  if b'data' in dict:
    dict[b'data'] = dict[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
  return dict

train = unpickle("train")
x_train = train[b'data']
y_train = train[b'fine_labels']
coarse = train[b'coarse_labels']
coarse = np.asarray(coarse)
# coarse14 is the set of indices in training dataset corresponding to human
coarse14 = []
for i in range(len(coarse)):
  if coarse[i] == 14:
    coarse14.append(i)
coarse0 = []
for i in range(len(coarse)):
  if coarse[i] == 0:
    coarse0.append(i)
coarse2 = []
for i in range(len(coarse)):
  if coarse[i] == 2:
    coarse2.append(i)
coarse6 = []
for i in range(len(coarse)):
  if coarse[i] == 6:
    coarse6.append(i)
coarse7 = []
for i in range(len(coarse)):
  if coarse[i] == 7:
    coarse7.append(i)

train_index = coarse0 + coarse2 + coarse6 + coarse7 + coarse14

#print(coarse14)
# coarse11 is the set of indices in training dataset corresponding to "large omnivores and herbivores"
#coarse11 = []
#for i in range(len(coarse)):
#  if coarse[i] == 11:
#    coarse11.append(i)
#print(coarse11)
#train_index = []
#for i in range(len(coarse)):
#	for j in range(4):
#		if coarse[i] == j:
#			train_index.append(i)
#train_index += coarse14

for i in coarse0:
  coarse[i] = 0

for i in coarse2:
  coarse[i] = 1

for i in coarse6:
  coarse[i] = 2

for i in coarse7:
  coarse[i] = 3

for i in coarse14:
  coarse[i] = 4

x_train = x_train[train_index]
y_train = coarse[train_index]


test = unpickle("test")
x_test = test[b'data']
coarse = test[b'coarse_labels']
coarse = np.asarray(coarse)
# coarse14 is the set of indices in training dataset corresponding to human
coarse14 = []
for i in range(len(coarse)):
  if coarse[i] == 14:
    coarse14.append(i)

coarse0 = []
for i in range(len(coarse)):
  if coarse[i] == 0:
    coarse0.append(i)
coarse2 = []
for i in range(len(coarse)):
  if coarse[i] == 2:
    coarse2.append(i)
coarse6 = []
for i in range(len(coarse)):
  if coarse[i] == 6:
    coarse6.append(i)
coarse7 = []
for i in range(len(coarse)):
  if coarse[i] == 7:
    coarse7.append(i)

test_index = coarse0 + coarse2 + coarse6 + coarse7 + coarse14

for i in coarse0:
  coarse[i] = 0

for i in coarse2:
  coarse[i] = 1

for i in coarse6:
  coarse[i] = 2

for i in coarse7:
  coarse[i] = 3

for i in coarse14:
        coarse[i] = 4


x_test = x_test[test_index]
y_test = coarse[test_index]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


def clip(tensor, sess):
  value = sess.run(tensor)
  np.clip(value, 0, 10, out=value)
  assign_op = tensor.assign(value)
  sess.run(assign_op)

def clipAll(tensorList, sess):
  for tensor in tensorList:
    clip(tensor, sess)





# input placeholder
x = tf.placeholder("float", shape=[None, 32, 32, 3], name="x")
lrt = tf.placeholder("float", shape=[])


# original model
with tf.variable_scope("HALF_MODEL", reuse=tf.AUTO_REUSE):
        # Create the network
        conv1_w = tf.get_variable(name = "conv1_W", shape = [3, 3, 3, 64], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
        conv1_b = tf.get_variable(name= "conv1_B", initializer = tf.constant(0.01, shape = [64]), trainable = False)
        _conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding="SAME") + conv1_b


        conv2_w = tf.get_variable(name = "conv2_W", shape = [3, 3, 64, 64], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
        conv2_b = tf.get_variable(name= "conv2_B", initializer = tf.constant(0.01, shape = [64]), trainable = False)
        _conv2 = tf.nn.conv2d(_conv1, conv2_w, strides=[1, 1, 1, 1], padding="SAME") + conv2_b

        _pool1 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


        conv3_w = tf.get_variable(name = "conv3_W", shape = [3, 3, 64, 128], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
        conv3_b = tf.get_variable(name= "conv3_B", initializer = tf.constant(0.01, shape = [128]), trainable = False)
        _conv3 = tf.nn.conv2d(_pool1, conv3_w, strides=[1, 1, 1, 1], padding="SAME") + conv3_b


        conv4_w = tf.get_variable(name = "conv4_W", shape = [3, 3, 128, 128], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
        conv4_b = tf.get_variable(name= "conv4_B", initializer = tf.constant(0.01, shape = [128]), trainable = False)
        _conv4 = tf.nn.conv2d(_conv3, conv4_w, strides=[1, 1, 1, 1], padding="SAME") + conv4_b
        _pool2 = tf.nn.max_pool(_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        _flattened = tf.reshape(_pool2, [-1, 8 * 8 * 128])

        fc1_w = tf.get_variable(name = "fc1_W", shape = [8*8*128, 1000], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
        fc1_b = tf.get_variable(name= "fc1_B", initializer = tf.constant(0.01, shape = [1000]), trainable = False)
        _fc1 = tf.matmul(_flattened, fc1_w) + fc1_b

        fc2_w = tf.get_variable(name = "fc2_W", shape = [1000, 1000], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
        fc2_b = tf.get_variable(name= "fc2_B", initializer = tf.constant(0.01, shape = [1000]), trainable = False)
        _fc2 = tf.matmul(_fc1, fc2_w) + fc2_b

        fc3_w = tf.get_variable(name = "fc3_W", shape = [1000, 5], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
        fc3_b = tf.get_variable(name= "fc3_B", initializer = tf.constant(0.01, shape = [5]), trainable = False)
        original_logits = tf.matmul(_fc2, fc3_w) + fc3_b


with tf.variable_scope("HALF_MODEL", reuse=tf.AUTO_REUSE):
	# Create the network
	conv1_w = tf.get_variable(name = "conv1_W", shape = [3, 3, 3, 64], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
	conv1_b = tf.get_variable(name= "conv1_B", initializer = tf.constant(0.01, shape = [64]), trainable = False)
	conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding="SAME") + conv1_b

	lambda1 = tf.get_variable(name = 'lambda1', initializer = tf.constant(1.0, shape=[64]))
	conv1_ = tf.multiply(conv1, tf.abs(lambda1))

	conv2_w = tf.get_variable(name = "conv2_W", shape = [3, 3, 64, 64], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
	conv2_b = tf.get_variable(name= "conv2_B", initializer = tf.constant(0.01, shape = [64]), trainable = False)
	conv2 = tf.nn.conv2d(conv1_, conv2_w, strides=[1, 1, 1, 1], padding="SAME") + conv2_b
	lambda2 = tf.get_variable(name = 'lambda2', initializer = tf.constant(1.0, shape=[64]))
	conv2_ = tf.multiply(conv2, tf.abs(lambda2))


	pool1 = tf.nn.max_pool(conv2_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


	conv3_w = tf.get_variable(name = "conv3_W", shape = [3, 3, 64, 128], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
	conv3_b = tf.get_variable(name= "conv3_B", initializer = tf.constant(0.01, shape = [128]), trainable = False)
	conv3 = tf.nn.conv2d(pool1, conv3_w, strides=[1, 1, 1, 1], padding="SAME") + conv3_b
	lambda3 = tf.get_variable(name = 'lambda3', initializer = tf.constant(1.0, shape=[128]))
	conv3_ = tf.multiply(conv3, tf.abs(lambda3))

	conv4_w = tf.get_variable(name = "conv4_W", shape = [3, 3, 128, 128], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
	conv4_b = tf.get_variable(name= "conv4_B", initializer = tf.constant(0.01, shape = [128]), trainable = False)
	conv4 = tf.nn.conv2d(conv3_, conv4_w, strides=[1, 1, 1, 1], padding="SAME") + conv4_b
	lambda4 = tf.get_variable(name = 'lambda4', initializer = tf.constant(1.0, shape=[128]))
	conv4_ = tf.multiply(conv4, tf.abs(lambda4))

	pool2 = tf.nn.max_pool(conv4_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

	flattened = tf.reshape(pool2, [-1, 8 * 8 * 128])

	fc1_w = tf.get_variable(name = "fc1_W", shape = [8*8*128, 1000], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
	fc1_b = tf.get_variable(name= "fc1_B", initializer = tf.constant(0.01, shape = [1000]), trainable = False)
	fc1 = tf.matmul(flattened, fc1_w) + fc1_b

	lambda5 = tf.get_variable(name = 'lambda5', initializer = tf.constant(1.0, shape=[1000]))
	fc1_ = tf.multiply(fc1, tf.abs(lambda5))

	fc2_w = tf.get_variable(name = "fc2_W", shape = [1000, 1000], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
	fc2_b = tf.get_variable(name= "fc2_B", initializer = tf.constant(0.01, shape = [1000]), trainable = False)
	fc2 = tf.matmul(fc1_, fc2_w) + fc2_b

	lambda6 = tf.get_variable(name = 'lambda6', initializer = tf.constant(1.0, shape=[1000]))
	fc2_ = tf.multiply(fc2, tf.abs(lambda6))

	fc3_w = tf.get_variable(name = "fc3_W", shape = [1000, 5], initializer = tf.truncated_normal_initializer(stddev = 0.01), trainable = False)
	fc3_b = tf.get_variable(name= "fc3_B", initializer = tf.constant(0.01, shape = [5]), trainable = False)
	lambda_logits = tf.matmul(fc2_, fc3_w) + fc3_b


	# Compute cross entropy as our loss function
	#softmax = tf.nn.softmax(lambda_logits)
	original_logits = tf.nn.softmax(original_logits)
	#xent = -tf.reduce_sum(original_logits * tf.log(softmax), 1)
	xent = tf.nn.softmax_cross_entropy_with_logits(logits=lambda_logits, labels=original_logits)
	regularization_penalty = tf.reduce_sum(lambda1) + tf.reduce_sum(lambda2) + tf.reduce_sum(lambda3) + tf.reduce_sum(lambda4) + tf.reduce_sum(lambda5) + tf.reduce_sum(lambda6)
	#l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.05, scope=None) 
	#weights = tf.trainable_variables()
	#regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
	regularized_loss = xent + 0.05 * regularization_penalty
	train_step = tf.train.MomentumOptimizer(lrt,0.9, use_nesterov=True).minimize(regularized_loss)

	correct_prediction = tf.equal(tf.argmax(lambda_logits, 1), tf.argmax(original_logits, 1))
	representation = tf.concat([lambda1, lambda2, lambda3, lambda4, lambda5, lambda6],0)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver({"conv1_W": conv1_w, "conv1_B": conv1_b, "conv2_W": conv2_w, "conv2_B": conv2_b, "conv3_W": conv3_w, "conv3_B": conv3_b, "conv4_W": conv4_w, "conv4_B": conv4_b, "fc1_W": fc1_w, "fc1_B": fc1_b, "fc2_W": fc2_w, "fc2_B": fc2_b, "fc3_W": fc3_w, "fc3_B": fc3_b})


saver.restore(sess, "/home/boyuanfeng/DGR/Old/coarseLabel.ckpy")
print("Model restored")

def initializeLambda(sess, lambda1_, lambda2_,lambda3_, lambda4_, lambda5_, lambda6_):
  assign_op1 = lambda1_.assign(np.ones(64))
  assign_op2 = lambda2_.assign(np.ones(64))
  assign_op3 = lambda3_.assign(np.ones(128))
  assign_op4 = lambda4_.assign(np.ones(128))
  assign_op5 = lambda5_.assign(np.ones(1000))
  assign_op6 = lambda6_.assign(np.ones(1000))
  sess.run(assign_op1)
  sess.run(assign_op2)
  sess.run(assign_op3)
  sess.run(assign_op4)
  sess.run(assign_op5)
  sess.run(assign_op6)

def count_nonzero(numList):
  count = 0
  for i in range(len(numList)):
    if numList[i] > 0.001:
      count += 1
  return count

# Train for DGR
def DGR_Training(x_, sess):
  x_ = np.reshape(x_, (1, 32, 32, 3))
  encoding = [1.0] * 2384
  xen = 0
  regularizer = 0
  lowestLoss = float("inf")
  tensorList = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6]

  for epoch in range(30):
    lr = 0.1
    #if epoch % 10 == 0:
    #  print(epoch)
    clipAll(tensorList, sess)
    #sess.run(train_step, feed_dict={x: x_, lrt: lr})
    [correct, loss, newEncoding, xen, regularizer] = sess.run([correct_prediction, regularized_loss, representation, xent, regularization_penalty], feed_dict={x: x_, lrt: lr})
    sess.run(train_step, feed_dict={x: x_, lrt: lr})
    print(newEncoding)
    print("xent: %f, regularization_penalty: %f" % (xen, regularizer))
    if correct == 1:
      if loss < lowestLoss:
        encoding = newEncoding
  #for j in range(len(encoding)):
  #  if encoding[j] == 0:
  #    print(j)
  for i in range(len(encoding)):
    if encoding[i] < 0.001:
      encoding[i] = 0
  print("non-zero number in encoding: %d" % count_nonzero(encoding))
  print(encoding)
  return encoding

x_test = test[b'data']
x_test = x_test.astype('float32')
store = np.empty([500, 2384])

import scipy

def analysis(store):
  correlation = []
  for j in range(len(store)):
    correlation.append(pearsonr(store[0], store[j])[0])
  np.save("/home/boyuanfeng/ClassEffect/DGR/correlation.npy", correlation)
  print("mean correlation: %f, std: %f" % (np.mean(correlation), np.var(correlation)))
  print(correlation)

#DGR_Training(x_test[coarse0[1]], sess)


for i in range(10):
  encoding = DGR_Training(x_test[coarse0[i]], sess)
  initializeLambda(sess, lambda1, lambda2,lambda3, lambda4, lambda5, lambda6)
  print(encoding)
  #encoding = encoding.tolist()
  store[i] = encoding
  print("%dth round:" % i)
  #print(type(encoding))
  #print(y_test[i])
print(store)
analysis(store)


print(tf.trainable_variables())

