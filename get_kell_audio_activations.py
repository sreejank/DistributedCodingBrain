"""
Code to get the branched network in Kell et al 2018 Neuron to "listen" to the sherlock episode and calculate the model's
Representational Similarity Matrix. Activity is obtained for each TR by calculating cochleagrams for TR-segment audio clips
and feeding those cochleagrams into the model.  
"""
import sys 
from branched_network_class import branched_network
import tensorflow as tf
import scipy.io.wavfile as wav

from pycochleagram import cochleagram as cgram 
from PIL import Image

from branched_network_class import branched_network
import tensorflow as tf
import scipy.io.wavfile as wav

from pycochleagram import cochleagram as cgram 
from PIL import Image

## Some helper functions. Code adapted from code in the repo: https://github.com/mcdermottLab/kelletal2018
def resample(example, new_size):
    im = Image.fromarray(example)
    resized_image = im.resize(new_size, resample=Image.ANTIALIAS)
    return np.array(resized_image)

def generate_cochleagram(wav_f, sr):
    # define parameters
    n, sampling_rate = 50, 16000
    low_lim, hi_lim = 20, 8000
    sample_factor, pad_factor, downsample = 4, 2, 200
    nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'
    strict = True

    # create cochleagram
    c_gram = cgram.cochleagram(wav_f, sr, n, low_lim, hi_lim, 
                               sample_factor, pad_factor, downsample,
                               nonlinearity, fft_mode, ret_mode, strict)
    
    # rescale to [0,255]
    c_gram_rescaled =  255*(1-((np.max(c_gram)-c_gram)/np.ptp(c_gram)))
    
    # reshape to (256,256)
    c_gram_reshape_1 = np.reshape(c_gram_rescaled, (211,400))
    c_gram_reshape_2 = resample(c_gram_reshape_1,(256,256))
    

    # prepare to run through network -- i.e., flatten it
    c_gram_flatten = np.reshape(c_gram_reshape_2, (1, 256*256)) 
    
    return c_gram_flatten

reverse=1

#You
sr, wav_f = wav.read(movie_audio_path) # note the sampling rate is 16000hz.





model=branched_network()

layers=[model.conv1,model.conv2,model.conv3,model.conv4_W,model.conv5_W,model.fc6_W,model.fc7_W,model.conv4_G,model.conv5_G,model.fc6_G,model.fc7_G]






activations=[]

samps=[]
for i in range(0,wav_f.shape[0],int(tr)):
    samps.append((i,i+1))
samps.append((samps[-1][1],len(wav_f)))

samps=samps[:-3]

for samp in samps:
	print(samp)
	#tf.reset_default_graph()
	c_gram=generate_cochleagram(wav_f[samp[0]:samp[1]],sr)
	activity=model.session.run(layers,feed_dict={model.x:c_gram})
	activations.append(activity)


names=['conv1','conv2','conv3','conv4_W','conv5_W','fc6_W','fc7_W','conv4_G','conv5_G','fc6_G','fc7_G']

activations=np.asarray(activations)

os.mkdir(data_dir+"activations/")
for i in range(len(names)):
	averaged=activations[:,i]
	name=names[i]
	full_name='kell_'+name
	np.save('activations/'+full_name+'.npy',averaged)

	rsm=np.corrcoef(averaged)
	#Only use RSM entries that are 10 TR's from the diagonal to avoid using data with high autocorrelation. 
    vec=rsm[np.triu(np.ones(rsm.shape),k=10).astype('bool')] 
	np.save(full_name+"_rsm.npy",vec)
