"""
Convert fMRI data from anatomical space to functional space using Shared Response Modeling. Outputs [Voxel,neighbors] matrix where neighbors corresponds to 
number of nearest neighbors to use in a searchlight in functional space. 
""" 
from cfg import *
from sklearn.neighbors import NearestNeighbors

def getStackedImages(fnames):
	images = io.load_images(fnames)
	
	
	masked_images = image.mask_images(images, brain_mask)
	images_concatenated = image.MaskedMultiSubjectData.from_masked_images(masked_images, len(fnames))
	images_concatenated[np.isnan(images_concatenated)] = 0

	return images_concatenated



brain_mask = io.load_boolean_mask("3mm_mask.nii.gz")

images_concatenated_train = getStackedImages(fnames_train)
train_data = [images_concatenated_train[:,:,i] for i in range(images_concatenated_train.shape[2])]
images_concatenated_test=getStackedImages(fnames_test)
test_data = [images_concatenated_test[:,:,i] for i in range(images_concatenated_test.shape[2])]

#dimension of functional space. 
features = DIM_FUNCTIONAL_SPACE
n_iter = 20

srm = brainiak.funcalign.srm.SRM(n_iter=n_iter,features=features)
srm.fit(train_data)
srm_weights=np.asarray(srm.w_) 


def run_subject(i):
	#Throw away any voxels that don't contribute to shared space at all (all 0 weights)
	X=srm_weights[i,:,:]
	good_voxels=(np.sum(X==0,axis=1)!=dim)
	good_indices=np.arange(X.shape[0])[good_voxels]
	bad_voxels=(np.sum(X==0,axis=1)==dim)
	bad_indices=np.arange(X.shape[0])[bad_voxels]
	short_X=X[good_voxels]
	
	#Leaf size is a parameter for the nearest neighbor tree data structure that makes nearest neighbors search faster given more memory. 
	nn=NearestNeighbors(n_neighbors=342,leaf_size=2000,metric='cosine') 
	nn.fit(short_X)
	short_space=nn.kneighbors(return_distance=False)  

	space=np.zeros((X.shape[0],N_FUNCTIONAL_NEIGHBORS))
	space[:,0]=np.arange(X.shape[0])
	space[good_voxels,1:]=good_indices[short_space]
	space[bad_voxels,:]=-1.0

	np.save(func_d+"space_"+str(i),space)

[run_subject(i) for i in range(16)]










