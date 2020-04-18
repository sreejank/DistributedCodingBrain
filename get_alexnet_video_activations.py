"""
Script to have AlexNet "watch" Sherlock episode and get the model's TR-by-TR Representational Similarity Matrix. Individual frames are fed
to the model and model activity of frames within each TR is averaged. 
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from alexnet import AlexNet
import keras.preprocessing.image as kpimage



x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)


model = AlexNet(x, keep_prob, 1000, [],path_to_weights='bvlc_alexnet.npy')

#Choose layer you want to get the activations for
score = model.fc6



activations_averaged=[]


trs=0
i=0
images=[]
print("Getting frames")
vidcap=cv2.VideoCapture(movie_video_path) #Edit in cfg for the movie video path. 
success,image=vidcap.read()
count=0
success=True
while success:
	cv2.imwrite("frames/frame"+str(count)+".jpg",image)
	success,image=vidcap.read()
	print("Read a new frame: "+str(success))
	count+=1
print("Done")


activations=[]
with tf.Session() as sess:
	
	
	sess.run(tf.global_variables_initializer())
	model.load_initial_weights(sess)
	
	
	i=0
	sz=35655
	while i<sz:
		fname="frames/frame"+str(i)+".jpg"
		image_raw= kpimage.load_img(fname, target_size=(227, 227, 3))
		image = kpimage.img_to_array(image_raw)
		img=np.asarray(image)
		img = img.reshape((1,227,227,3))
		lay = sess.run(score, feed_dict={x: img, keep_prob: 1})
		activations.append(lay)
		i+=1




activations=np.asarray(activations)
x=activations

averaged=[]
i=0
tr=0
"""
Frames within the same TR are averaged. Theoretically, there are 37.5 frames in a TR. To prevent autocorrelation of
activity across adjacent TR's, we average the middle 27 frames of each TR. 
"""
while tr<1030:
    interval=(tr*37.5,(tr+1)*37.5)
    lst=range(int(interval[0])+2,int(interval[1])-2)
    idx=lst[:27]
    averaged.append(np.average(x[idx,0,:],axis=0))
    tr+=1	
averaged=np.asarray(averaged)

np.save(data_dir+'fc6_activity.npy',averaged)

rsm=np.corrcoef(averaged)
#Only use RSM entries that are 10 TR's from the diagonal to avoid using data with high autocorrelation. 
vec=rsm[np.triu(np.ones(rsm.shape),k=TR_BUFFER_SIZE).astype('bool')] 
np.save(data_dir+"fc6_rsm.npy",vec)





