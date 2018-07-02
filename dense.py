import numpy as np
import tensorflow as tf
import random
import keras
from sklearn.utils import shuffle
import pickle

AllGateVariable = {}


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
  return v


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable_msra([ kernel_size, kernel_size, in_features, out_features ], name = 'kernel')
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  # Use gate to select Data Routing Path
  gate = gate_variable(out_features)
  conv = tf.multiply(conv, tf.abs(gate))
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

def batch_activ_conv(current, in_features, out_features, kernel_size, is_training, keep_prob):
  with tf.variable_scope("composite_function", reuse = tf.AUTO_REUSE):
    current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
    current = tf.nn.relu(current)
    current = conv2d(current, in_features, out_features, kernel_size)
    current = tf.nn.dropout(current, keep_prob)
  return current

def block(input, layers, in_features, growth, is_training, keep_prob):
  current = input
  features = in_features
  for idx in range(layers):
    with tf.variable_scope("layer_%d" % idx, reuse = tf.AUTO_REUSE):
      tmp = batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
      current = tf.concat((current, tmp), 3)
      features += growth
  return current, features

def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'SAME')

# shuffle function is here, used for traning time
def shuffle_images_and_labels(images, labels):
  rand_indexes = np.random.permutation(images.shape[0])
  shuffled_images = images[rand_indexes]
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


def clip(tensor, sess):
  value = sess.run(tensor)
  np.clip(value, -10, 10, out=value)
  assign_op = tensor.assign(value)
  sess.run(assign_op)

def clipAll(tensorList, sess):
  for tensor in tensorList:
    clip(tensor, sess)


def run_model(data, label_count, depth):
  tf.reset_default_graph()
  weight_decay = 0.05
  layers = (depth - 4) // 3
  graph = tf.Graph()
  with graph.as_default():
    xs = tf.placeholder("float", shape=[None, 32, 32, 3])
    ys = tf.placeholder("float", shape=[None, label_count])
    lr = tf.placeholder("float", shape=[])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder("bool", shape=[])

    with tf.variable_scope("Initial_convolution", reuse = tf.AUTO_REUSE):
      current = conv2d(xs, 3, 16, 3)
    with tf.variable_scope("Block_1", reuse = tf.AUTO_REUSE):
      current, features = block(current, layers, 16, 12, is_training, keep_prob)
    with tf.variable_scope("Transition_after_block_1", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
      current = avg_pool(current, 2)
    with tf.variable_scope("Block_2", reuse = tf.AUTO_REUSE):
      current, features = block(current, layers, features, 12, is_training, keep_prob)
    with tf.variable_scope("Transition_after_block_2", reuse = tf.AUTO_REUSE):
      current = batch_activ_conv(current, features, features, 1, is_training, keep_prob)
      current = avg_pool(current, 2)
    with tf.variable_scope("Block_3", reuse = tf.AUTO_REUSE):
      current, features = block(current, layers, features, 12, is_training, keep_prob)
    with tf.variable_scope("Transition_to_classes", reuse = tf.AUTO_REUSE):
      current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None, trainable=False)
      current = tf.nn.relu(current)
      current = avg_pool(current, 8)
      final_dim = features
      current = tf.reshape(current, [ -1, final_dim ])
      Wfc = weight_variable_xavier([ final_dim, label_count ], name = 'W')
      bfc = bias_variable([ label_count ])
      ys_ = tf.matmul(current, Wfc) + bfc
    prediction = tf.nn.softmax(ys_)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = ys, logits = ys_))
    #l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in AllGateVariable.values()])
    #l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.05)
    #l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())
    

    l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in AllGateVariable.values()])
    l1_loss = l1_loss * 0.5
    total_loss = l1_loss + cross_entropy
    #total_loss = l1_loss + cross_entropy
    #total_loss = l2_loss*weight_decay + cross_entropy
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
      variable_parametes = 1
      for dim in shape:
          variable_parametes *= dim.value
      total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e3))
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
    saver.restore(session, "checkpoints_2classes/augmentation.ckpt-300")
    print("restored successfully!")
    # Train DGR
    generatedGate = []
    uniqueGate = []
    train_data = train_data_normalization
    train_labels = data['train_labels']
    train_labels = session.run(prediction, feed_dict = {xs: train_data, ys: train_labels, lr: learning_rate, is_training: False, keep_prob: 1.0 })
      
    for epoch in range(1, 300+1):
      #clipAll(AllGateVariable.values(), session)
      #if epoch == 75: learning_rate /= 10
      #if epoch == 150: learning_rate /= 10
      #if epoch == 150: learning_rate /= 10
      
      batch_res = session.run([ train_step, accuracy, correct_prediction, ys_, l1_loss, cross_entropy, top5],
          feed_dict = { xs: train_data, ys: train_labels, lr: learning_rate, is_training: False, keep_prob: 1.0 })
      print("Epoch" + str(epoch)+ ", correct prediction: "+str(batch_res[2])+", xent: "+str(batch_res[5])+", l1_loss: "+str(batch_res[4]) + ", top5: " + str(batch_res[6]))
      newGate = []
      for gate in AllGateVariable.values():
        tmp = gate.eval()
        newGate.append(list(tmp))
      
      if batch_res[1] > 0.99:
        #print("epoch"+str(epoch)+" is successful. Accuracy is "+str(batch_res[1]))
        #print(newGate)
        generatedGate = list(newGate)
        #uniqueGate += list(newGate)

      '''
      for gate in AllGateVariable.values():
        tmp = gate.eval()
        print("Original allGates = pickle.load(open('allGates.p','rb'))Value"+str(tmp))
        for idx in range(len(tmp)):
          if tmp[idx] > 10:
            tmp[idx] = 10
          if tmp[idx] < -10:
            tmp[idx] = -10
        assign_op = gate.assign(tmp)
        session.run(assign_op)  # or `assign_op.op.run()`
        print("Value after clip")
        print(gate.eval()) 
      '''

    #print("The generatedGate is: ")
    #print(generatedGate)   pickle.dump(allGates, open("allGates.p",'wb'))
    return generatedGate


def run():
  label_count = 100
  targetList = range(1)
  train_data, train_labels = generateSpecializedData('train', targetList, 10, 0)
  test_data, test_labels = generateSpecializedData("test", targetList, 1, 0)
  print ("Train:", np.shape(train_data), np.shape(train_labels))
  print ("Test:", np.shape(test_data), np.shape(test_labels))
  allGates = []
  for i in range(10):
    print("Image "+str(i))
    print(train_data[i].shape)
    data = { 'train_data': train_data[i].reshape((1,32,32,3)),
        'train_labels': train_labels[i].reshape(1,100),
        'test_data': train_data[i].reshape((1,32,32,3)),
        'test_labels': train_labels[i] }
    generatedGate = run_model(data, label_count, 40)
    allGates.append(generatedGate)
  pickle.dump(allGates, open("allGates200.p",'wb'))


run()
