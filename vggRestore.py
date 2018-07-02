import numpy as np
import tensorflow as tf
import random
import keras
import pickle
from sklearn.utils import shuffle
from decimal import *

getcontext().prec = 7


AllGateVariable = {}
AllGateVariableValue = []
convValues = []

def setzero(x):
  for i in range(len(x)):
    if x[i] < 0.01:
      x[i] = 0
  return x

def unpickle(file):
  import pickle
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding = 'bytes')
  fo.close()
  if b'data' in dict:
    dict[b'data'] = dict[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
  return dict

def load_data_one(f):
  batch = unpickle(f)
  data = batch[b'data']
  labels = batch[b'fine_labels']
  return data, labels

def load_data(files, data_dir, label_count):
  data, labels = load_data_one(data_dir + '/' + files[0])
  for f in files[1:]:
    data_n, labels_n = load_data_one(data_dir + '/' + f)
    data = np.append(data, data_n, axis=0)
    labels = np.append(labels, labels_n, axis=0)
  labels = np.array([ [ float(i == label) for i in range(label_count) ] for label in labels ])
  return data, labels

def run_in_batch_avg(session, tensors, batch_placeholders, feed_dict={}, batch_size=200):                              
  res = [ 0 ] * len(tensors)                                                                                           
  batch_tensors = [ (placeholder, feed_dict[ placeholder ]) for placeholder in batch_placeholders ]                    
  total_size = len(batch_tensors[0][1])                                                                                
  batch_count = (total_size + batch_size - 1) // batch_size                                                             
  for batch_idx in range(batch_count):                                                                                
    current_batch_size = None                                                                                          
    for (placeholder, tensor) in batch_tensors:                                                                        
      batch_tensor = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                                         
      current_batch_size = len(batch_tensor)                                                                           
      feed_dict[placeholder] = tensor[ batch_idx*batch_size : (batch_idx+1)*batch_size ]                               
    tmp = session.run(tensors, feed_dict=feed_dict)                                                                    
    res = [ r + t * current_batch_size for (r, t) in zip(res, tmp) ]                                                   
  return [ r / float(total_size) for r in res ]

def weight_variable_msra(shape, name):
  return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.variance_scaling_initializer(), trainable=False)

def weight_variable_xavier(shape, name):
  return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), trainable=False)

def bias_variable(shape, name = 'bias'):
  initial = tf.constant(0.0, shape = shape)
  return tf.get_variable(name = name, initializer = initial, trainable=False)

def gate_variable(length, name = 'gate'):
  initial = tf.constant([1.0] * length)
  v = tf.get_variable(name = name, initializer = initial)
  AllGateVariable[v.name] = v
  v = tf.abs(v)
  v = v - tf.constant([0.01]*length)
  v = tf.nn.relu(v)
  AllGateVariableValue.append(v)
  return v



def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable_msra([ kernel_size, kernel_size, in_features, out_features ], name = 'kernel')
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  gate = gate_variable(out_features)
  conv = tf.multiply(conv, tf.abs(gate))
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
  with tf.variable_scope("composite_function", reuse = tf.AUTO_REUSE):
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
    convValues.append(current)
    current = tf.nn.relu(current)
    #current = tf.nn.dropout(current, keep_prob)
  return current


def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'SAME')

# shuffle function is here, used for traning time
def shuffle_images_and_labels(images, labels):
  #print(images.shape)
  #print(images.shape[0])
  #print(type(images.shape[0]))
  rand_indexes = np.random.permutation(images.shape[0])
  shuffled_images = images[rand_indexes]
  #print(type(labels))
  #print(labels.shape)
  shuffled_labels = labels[rand_indexes]
  return shuffled_images, shuffled_labels

# calculate the means and stds for the whole dataset per channel
def measure_mean_and_std(images):
  means = []
  stds = []
  for ch in range(images.shape[-1]):
    means.append(np.mean(images[:, :, :, ch]))
    stds.append(np.std(images[:, :, :, ch]))
  return means, stds

# normalization for per channel
def normalize_images(images):
  images = images.astype('float64')
  means, stds = measure_mean_and_std(images)
  for i in range(images.shape[-1]):
    images[:, :, :, i] = ((images[:, :, :, i] - means[i]) / stds[i])
  return images

# data augmentation, contains padding, cropping and possible flipping
def augment_image(image, pad):
  init_shape = image.shape
  new_shape = [init_shape[0] + pad * 2, init_shape[1] + pad * 2, init_shape[2]]
  zeros_padded = np.zeros(new_shape)
  zeros_padded[pad:init_shape[0] + pad, pad: init_shape[1] + pad, :] = image
  init_x = np.random.randint(0, pad * 2)
  init_y = np.random.randint(0, pad * 2)
  cropped = zeros_padded[init_x: init_x + init_shape[0], init_y: init_y + init_shape[1], :]
  flip = random.getrandbits(1)
  if flip:
    cropped = cropped[:, ::-1, :]
  return cropped

def augment_all_images(initial_images, pad):
  new_images = np.zeros(initial_images.shape)
  for i in range(initial_images.shape[0]):
    new_images[i] = augment_image(initial_images[i], pad = 4)
  return new_images


def generateSpecializedData(fileName, target, count1, count2):
	train = unpickle(fileName)
	x_train = train[b'data']
	y_train = train[b'fine_labels']
	y_train = np.asarray(y_train)
	train_index = []
	for i in range(len(target)):
		index = list(np.where(y_train[:] == target[i])[0])[0:count1]
		train_index += index
	for i in range(0, 100):
		if i in target:
			continue
		index = list(np.where(y_train[:] == i)[0])[0:count2]
		train_index += index
	train_index = shuffle(train_index)
	sp_y = y_train[train_index]
	sp_x = x_train[train_index]
	sp_y = keras.utils.to_categorical(sp_y, 100)
	sp_y = sp_y.astype('float32')
	sp_x = sp_x.astype('float32')
	return sp_x, sp_y

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')

def batch_activ_fc(current, in_features, out_features, is_training):
  Wfc = weight_variable_xavier([ in_features, out_features ], name = 'W')
  bfc = bias_variable([ out_features ])
  current = tf.matmul(current, Wfc) + bfc
  gate = gate_variable(out_features)
  current = tf.multiply(current, tf.abs(gate))
  current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
  current = tf.nn.relu(current)
  return current


def run_model(data, label_count, depth):
  #tf.reset_default_graph()
  weight_decay = 0.05
  graph = tf.Graph()
  AllGateVariable.clear()
  AllGateVariableValue.clear()
  convValues.clear()


  with graph.as_default():
    xs = tf.placeholder("float", shape=[None, 32, 32, 3])
    ys = tf.placeholder("float", shape=[None, label_count])
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])
    penalty = tf.placeholder(tf.float32)

    with tf.variable_scope("Conv1", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(xs, 3, 64, 3, is_training, keep_prob)
    with tf.variable_scope("Conv2", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 64, 64, 3, is_training, keep_prob)
      current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv3", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 64, 128, 3, is_training, keep_prob)
    with tf.variable_scope("Conv4", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 128, 128, 3, is_training, keep_prob)
      current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv5", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 128, 256, 3, is_training, keep_prob)
    with tf.variable_scope("Conv6", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 256, 256, 3, is_training, keep_prob)
    with tf.variable_scope("Conv7", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 256, 256, 1, is_training, keep_prob)
      current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv8", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 256, 512, 3, is_training, keep_prob)
    with tf.variable_scope("Conv9", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 512, 512, 3, is_training, keep_prob)
    with tf.variable_scope("Conv10", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 512, 512, 1, is_training, keep_prob)
      current = maxpool2d(current, k=2)
    with tf.variable_scope("Conv11", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 512, 512, 3, is_training, keep_prob)
    with tf.variable_scope("Conv12", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 512, 512, 3, is_training, keep_prob)
    with tf.variable_scope("Conv13", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, 512, 512, 1, is_training, keep_prob)
      current = maxpool2d(current, k=2)
      current = tf.reshape(current, [ -1, 512 ])
    with tf.variable_scope("FC14", reuse = tf.AUTO_REUSE):
      current = batch_activ_fc(current, 512, 4096, is_training)
    with tf.variable_scope("FC15", reuse = tf.AUTO_REUSE):
      current = batch_activ_fc(current, 4096, 4096, is_training)
    with tf.variable_scope("FC16", reuse = tf.AUTO_REUSE):
      Wfc = weight_variable_xavier([ 4096, label_count ], name = 'W')
      bfc = bias_variable([ label_count ])
      ys_ = tf.matmul(current, Wfc) + bfc
    prediction = tf.nn.softmax(ys_)
    conv_value = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in convValues])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = ys_))
    l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in AllGateVariableValue])
    l1_loss = l1_loss * penalty
    total_loss = l1_loss + cross_entropy
    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(total_loss)
    correct_prediction = tf.equal(tf.argmax(ys_, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    top5 = tf.nn.in_top_k(predictions=ys_, targets=tf.argmax(ys, 1), k=5)
    top_5 = tf.reduce_mean(tf.cast(top5, 'float'))


  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(graph=graph, config = config) as session:
    total_parameters = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      #print(variable.name)
      variable_parametes = 1
      for dim in shape:
          variable_parametes *= dim.value
      total_parameters += variable_parametes
    print("Total training params: %.1fK" % (total_parameters / 1e3))


    batch_size = 64
    learning_rate = 0.1
    session.run(tf.global_variables_initializer())
    # normalization images here
    train_data_normalization = normalize_images(data['train_data'])
    test_data_normalization = normalize_images(data['test_data'])

    batch_count = len(train_data_normalization) // batch_size
    savedVariable = {}
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      variable = i
      name = i.name
      if name == 'pl:0':
        continue
      if name in AllGateVariable:
        continue 
      if len(name) >= 8 and name[-11:] == '/Momentum:0':
        name_prefix = name[:-11]
        name_prefix += ':0'
        if name_prefix in AllGateVariable:
          continue
      name = i.name[:-2]
      savedVariable[name] = variable
    saver = tf.train.Saver(savedVariable)
    #saver = tf.train.Saver(max_to_keep = None)
    saver.restore(session, "vggNet/augmentation.ckpt-120")
    print("restored successfully!")
    


    # Train DGR
    generatedGate = []
    uniqueGate = []
    train_data = train_data_normalization
    train_labels = data['train_labels']
    train_labels = session.run(prediction, feed_dict = {xs: train_data, ys: train_labels, lr: learning_rate, is_training: False, keep_prob: 1.0, penalty: 0.05 })
    #print(train_labels)
    lowest_L1 = 10000
    __p = 0.05
    for epoch in range(1, 100+1):
      #clipAll(AllGateVariable.values(), session)
      #if epoch == 75: learning_rate /= 10
      #if epoch == 150: learning_rate /= 10
      #if epoch == 150: learning_rate /= 10
      if epoch == 50: 
        learning_rate /= 10
        __p *= 10
      #if epoch == 75: learning_rate /= 10
      batch_res = session.run([ train_step, accuracy, correct_prediction, ys_, l1_loss, cross_entropy, top5, conv_value],
          feed_dict = { xs: train_data, ys: train_labels, lr: learning_rate, is_training: False, keep_prob: 1.0, penalty: __p })
      print("Epoch" + str(epoch)+ ", correct prediction: "+str(batch_res[2])+", xent: "+str(batch_res[5])+", l1_loss: "+str(batch_res[4]) + ", top5: " + str(batch_res[6]) + ", conv: "+str(batch_res[7]))
      newGate = []
      for gate in AllGateVariable.values():
        tmp = gate.eval()
        newGate.append(list(tmp))
      if batch_res[4] > lowest_L1:
        continue
      if batch_res[1] > 0.99 and batch_res[4] != 'nan' and batch_res[4] < 1000:
        generatedGate = list(newGate)
        lowest_L1 = batch_res[4]

    return generatedGate




def run():

  for idx in [2]:
    print("class"+str(idx))
    targetList = []
    targetList.append(idx)
    label_count = 100
    train_data, train_labels = generateSpecializedData('train', targetList, 50, 0)
    test_data, test_labels = generateSpecializedData("test", targetList, 1, 0)
    print ("Train:", np.shape(train_data), np.shape(train_labels))
    print ("Test:", np.shape(test_data), np.shape(test_labels))

    allGates = []
    for i in range(30):
      print("Image "+str(i))
      print(train_data[i].shape)
      data = { 'train_data': train_data[i].reshape((1,32,32,3)),
          'train_labels': train_labels[i].reshape(1,100),
          'test_data': train_data[i].reshape((1,32,32,3)),
          'test_labels': train_labels[i] }
      generatedGate = run_model(data, label_count, 40)
      allGates.append(generatedGate)
    saveFile = 'allGates'+str(idx)
    pickle.dump(allGates, open(saveFile,'wb'))


run()
