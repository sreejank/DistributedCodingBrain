"""
Make an fMRI volume of the functional searchlight data. Specify a subject and kernel to choose the result you want to
"""
from cfg import *

sub=subs[int(sys.argv[1])]
kernel=sys.argv[2]

fname_data=results_dir+sub+"_functional_SL_"+kernel+".npy"
output_name=results_dir+'/'+kernel+"/"+sub+'_whole_brain_functional_SL_warped.nii.gz'

mask=nib.load("3mm_mask.nii.gz").get_data()
mask_vec=mask.reshape(np.prod(mask.shape[:3]),)
data=np.load(fname_data)
full_data=np.zeros(mask_vec.shape)
full_data[mask_vec==1]=data
vol=full_data.reshape((mask.shape[0],mask.shape[1],mask.shape[2]))
affine_mat=nib.load(test_d+sub+"_test.nii.gz").affine
nii = nib.Nifti1Image(vol.astype('float32'), affine_mat)
nib.save(nii, output_name)


