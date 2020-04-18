from brainiak import image, io
import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVC
from scipy.stats.stats import pearsonr 
from sklearn.svm import LinearSVC
import sys
import os
from sklearn.metrics import accuracy_score
import brainiak.funcalign.srm
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from brainiak.fcma.util import compute_correlation

data_dir="test_sherlock/"
subs=['s1','s2','s3', 's4','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17']
raw_data_name=data_dir+"movie_files/sherlock_movie_"
fnames_data=[raw_data_name+s+".nii" for s in subs] 

if not os.path.exists(data_dir+"train/"):
	os.mkdir(data_dir+"train/")
if not os.path.exists(data_dir+"test/"):
	os.mkdir(data_dir+"test/")
if not os.path.exists(data_dir+"functional_space/"):
	os.mkdir(data_dir+"functional_space/")
if not os.path.exists(data_dir+"results/"):
	os.mkdir(data_dir+"results/")

train_d=data_dir+"train/"
test_d=data_dir+"test/"
func_d=data_dir+"functional_space/"
results_dir=data_dir+"results/"

fnames_train=[train_d+s+"_train.nii.gz" for s in subs]
fnames_test=[test_d+s+"_test.nii.gz" for s in subs]


movie_video_path=data_dir+"" #Edit the empty quotes with the stimulus video (mp4/m4v file) 
movie_audio_path=data_dir+"" #Edit the empty quotes with the stimulus audi (wav file). You can get this from the video by using ffmpeg. 

TR_BUFFER_SIZE=10
DIM_FUNCTIONAL_SPACE=200
N_FUNCTIONAL_NEIGHBORS=343
SL_ANAT_RAD=3











