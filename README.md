# DistributedCodingBrain
Code for my upcoming paper "Distributed Coding of Models and Features in the Human Brain" (in submission)

Instructions for use of the new method:

1) Make an empty directory for the data. This will be the directory that all of the data will be kept in.

2) Download the Sherlock fMRI dataset into that directory(https://dataspace.princeton.edu/jspui/handle/88435/dsp01nz8062179). This should create the folder "movie_files" within the directory you just made. 

3) Edit cfg.py and use the path of the directory you just made for the value of the variable "data_dir."

4) Run python split.py to split the data into two halves, one set used for training the Shared Response Model and one set used for the main analyses. 

5) Run python functional_space.py to convert the sherlock test data form anatomical space to functional space. 

6) To run a standard anatomical searchlight analysis described in the paper, run python anatomical_searchlight.py <subject_number> <kernel> <labels> where subject_number is a number from [0,15] indicating the subject you want to run, kernel is one of {'sensory_rsa','semantic_decoding'} indicating the analysis you want to run (similarity analysis with deep learning models and semantic vector decoding respectively), and labels is one of {'alexnet','kellnet'} if kernel is 'rsa' (to choose the deep learning model you want to use). If kernel is "decode" then choose avg_100dim_wordvec_mat_Sep12.npz as label.  
  
 7) To run a standard functional searchlight analysis described in the paper, run python functional_searchlight.py <subject_number> <kernel> <labels> in the same fashion as step 6. Make sure you do step 5 before doing this. 
  
 8) To see the functional searchlight results in anatomical space, run python warp_functional2anatomical.py <subject_number> <kernel> where subject_number is a number from [0,15] indicating the subject whose results you want to warp and kernel is one of {'sensory_rsa','semantic_decoding'} indicating the analysis you want to warp the results for. 
  
 The Representational Similarity Matrices (RSM's) for the deep learning models have been provided in the rsm folder as well as the labels. The code to calculate the RSM's can be found in "get_alexnet_video_activations.py" and "get_kell_audio_activations.py." The networks can be found in "alexnet.py" and "branched_network_class.py." 
