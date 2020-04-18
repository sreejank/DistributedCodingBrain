"""
Split data into training and testing half. The training data is used to fit the SRM and the test data is used for the analyses. 
"""
from cfg import *

for i,fname in enumerate(fnames_data):
	print("Subject "+subs[i])
	img=nib.load(fname)
	x=img.get_data()
	train_img=nib.Nifti1Image(x[:,:,:,:946],img.affine)
	test_img=nib.Nifti1Image(x[:,:,:,946:],img.affine)
	nib.save(train_img,train_d+subs[i]+"_train.nii.gz")
	nib.save(test_img,test_d+subs[i]+"_test.nii.gz")
