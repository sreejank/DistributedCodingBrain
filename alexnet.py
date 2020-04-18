
"""
Tensorflow implementation of Alexnet deep learning model. 
"""
import tensorflow as tf
import numpy as np

class AlexNet(object):

	def __init__(self,x,keep_prob,num_classes,skip_layer,path_to_weights='bvlc_alexnet.npy'):
		"""
		Inputs:
		x: tf.placeholder for input images
		keep_prob: tf.placeholder for dropout rate
		num_classes: int, number of classes of the new dataset
		skip_layer: list of layers (strings)  you want to reinitialize
		path_to_weights: path to the pretrained weights
		"""
		self.X=x
		self.NUM_CLASSES=num_classes
		self.KEEP_PROB=keep_prob
		self.SKIP_LAYER=skip_layer
		self.IS_TRAINING=False
		self.WEIGHTS_PATH=path_to_weights

		self.make()

	def make(self):
		#Layer 1
		conv1=conv(self.X,11,11,96,4,4,padding='VALID',name='conv1')
		self.conv1=conv1
		norm1=lrn(conv1,2,1e-05,0.75,name='norm1')
		pool1=max_pool(norm1,3,3,2,2,padding='VALID',name='pool1')
		self.pool1=tf.contrib.layers.flatten(pool1)
		#Layer 2
		conv2=conv(pool1,5,5,256,1,1,groups=2,name='conv2')
		self.conv2=conv2
		norm2=lrn(conv2,2,1e-05,0.75,name='norm2')
		pool2=max_pool(norm2,3,3,2,2,padding='VALID',name='pool2')
		self.pool2=tf.contrib.layers.flatten(pool2)

		#Layer 3
		conv3=conv(pool2,3,3,384,1,1,name='conv3')
		self.conv3=conv3
		
		#Layer 4
		conv4=conv(conv3,3,3,384,1,1,groups=2,name='conv4')
		self.conv4=conv4

		#Layer 5
		conv5=conv(conv4,3,3,256,1,1,groups=2,name='conv5')
		self.conv5=conv5
		pool5=max_pool(conv5,3,3,2,2,padding='VALID',name='pool5')

		#Layer 6
		self.flattened=tf.reshape(pool5,[-1,6*6*256])
		self.fc6=fc(self.flattened,6*6*256,4096,name='fc6')
		dropout6=dropout(self.fc6,self.KEEP_PROB)

		#Layer 7
		self.fc7=fc(dropout6,4096,4096,name='fc7')
		dropout7=dropout(self.fc7,self.KEEP_PROB)

		#Layer 8
		self.fc8=fc(dropout7,4096,self.NUM_CLASSES,relu=False,name='fc8')

		#self.layers=[[conv1,norm1,pool1],[conv2,norm2,pool2],[conv3],[conv4],[conv5,pool5],[fc6,dropout6],[self.fc7,dropout7],[self.fc8]]
	


	def load_initial_weights(self,session):
		weights_dict=np.load(self.WEIGHTS_PATH,encoding='bytes').item()
		for op_name in weights_dict:
			if op_name not in self.SKIP_LAYER:
				with tf.variable_scope(op_name,reuse=True):
					for data in weights_dict[op_name]:
						if len(data.shape)==1:
							var=tf.get_variable('biases',trainable=False)
							session.run(var.assign(data))
						else:
							var=tf.get_variable('weights',trainable=False)
							session.run(var.assign(data))


#Convolution layer
def conv(x,filter_height,filter_width,num_filters,stride_y,stride_x,name,padding='SAME',groups=1):
	#Number of input channels (example: RGB)
	input_channels=int(x.get_shape()[-1])

	#Convolve image i with weights w specified with filters of stride_y and stride_x. 
	convolve=lambda i,w: tf.nn.conv2d(i,w,strides=[1,stride_y,stride_x,1],padding=padding)

	#get weights/biases for the conv layer
	with tf.variable_scope(name) as scope:
		weights=tf.get_variable('weights',shape=[filter_height,filter_width,input_channels/groups,num_filters])
		biases=tf.get_variable('biases',shape=[num_filters])
	
	if groups==1:
		conv=convolve(x,weights)
	#If >1 groups, split up input/weights by groups and make a list of each paired convolution
	else:
		input_groups=tf.split(axis=3,num_or_size_splits=groups,value=x)
		weight_groups=tf.split(axis=3,num_or_size_splits=groups,value=weights)
		output_groups=[convolve(i,k) for i,k in zip(input_groups,weight_groups)]

		conv=tf.concat(axis=3,values=output_groups)

	biased_convolution=tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape().as_list())
	final_output=tf.nn.relu(biased_convolution,name=scope.name)
	return final_output

#Fully connected layer
def fc(x,num_in,num_out,name,relu=True):
	with tf.variable_scope(name) as scope:
		weights=tf.get_variable('weights',shape=[num_in,num_out],trainable=True)
		biases=tf.get_variable('biases',[num_out],trainable=True)

		output=tf.nn.xw_plus_b(x,weights,biases,name=scope.name)
		if relu:
			return tf.nn.relu(output)
		else:
			return output

#Max Pooling Layer
def max_pool(x,filter_height,filter_width,stride_y,stride_x,name,padding='SAME'):
	return tf.nn.max_pool(x,ksize=[1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],padding=padding,name=name)

#Local Response Normalization
def lrn(x,radius,alpha,beta,name,bias=1.0):
	return tf.nn.local_response_normalization(x,depth_radius=radius,alpha=alpha,beta=beta,bias=bias,name=name)

#Dropout
def dropout(x,keep_prob):
	return tf.nn.dropout(x,keep_prob)



	
							
