"""
Run a standard anatomical searchlight. Choose "rsa" or "classify" as the kernel to pick the analysis you want to run
and choose a subject you want to run on. 
"""
from cfg import *

sl_rad = SL_ANAT_RAD
max_blk_edge = 5
pool_size = 1

sub=int(sys.argv[1])
kernel=sys.argv[2] 
bcvarname=sys.argv[3]

if sub<0 or sub>=16:
    print("Invalid subject number")
    sys.exit(1)
sub=subs[sub]

if kernel!='sensory_rsa' and kernel!='semantic_decoding':
    print("Invalid kernel name")
    sys.exit(1)

if bcvarname=='alexnet':
    bcvarname='rsm/alex_fc6_rsm.npy'
    bcvar=np.load(bcvarname)
elif bcvarname=='kellnet':
    bcvarname='rsm/kell_conv4_W_rsm.npy'
    bcvar=np.load(bcvarname)
else:
    bcvarname='rsm/avg_100dim_wordvec_mat_Sep12.npz'
    bcvar=np.load(bcvarname)['vecs']
else:
    print("Invalid label or rsm")
    sys.exit(1) 



output_name = results_dir+'/'+kernel+"/"+sub+'_whole_brain_anatomical_SL.nii.gz'

# Get information
print("Loading data")
nii = nib.load(test_d+sub+"_test.nii.gz")
affine_mat = nii.affine 
dimsize = nii.header.get_zooms() 
data = nii.get_data()
mask=nib.load("3mm_mask.nii.gz").get_data()







# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

print("Preparing searchlight")
# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)

# Distribute the information to the searchlights (preparing it to run)
sl.distribute([data], mask)



# Broadcast variables
sl.broadcast(bcvar)

# Set up the kernel function

def rsa(data,mask,myrad,bcvar): 
    if np.sum(mask)<2:
        return 0.0

    data4D=data[0]
    mask=mask.astype('bool')
    bolddata_sl=data4D[mask,:].T

    alexnet=bcvar
    human=np.corrcoef(bolddata_sl[:,:])
    vec=human[np.triu(np.ones(human.shape),k=10).astype('bool')]
    return pearsonr(vec,alexnet)[0] 
 
def decode(data,mask,myrad,bcvar):
    if np.sum(mask)<2:
        return 0.0

    data4D=data[0]
    #bolddata_sl=data4D.reshape(mask.shape[0]*mask.shape[1]*mask.shape[2],data[0].shape[3]).T
    mask=mask.astype('bool')
    bolddata_sl=data4D[mask,:].T

    model=Ridge() 
    y=bcvar
    
    split_idx=530
    train_X=bolddata_sl[:split_idx,:]
    train_y=y[:split_idx,:] 

    test_X=bolddata_sl[split_idx:,:]
    test_y=y[split_idx:,:]

    model.fit(train_X,train_y)
    pred_y=model.predict(test_X)
    
    print(pred_y.shape,test_y.shape)
    true_chunks=np.asarray([np.hstack(subarray) for subarray in np.array_split(test_y,25)])
    pred_chunks=np.asarray([np.hstack(subarray) for subarray in np.array_split(pred_y,25)])
    corr_mtx=compute_correlation(pred_chunks,true_chunks)

    max_idx =  np.argmax(corr_mtx, axis=1)
    ans=sum(max_idx == range(25)) / 25.0
    print(ans)
    return ans




if kernel=='sensory_rsa':
    f=rsa
elif kernel=='semantic_decoding':
    f=decode


sl_result = sl.run_searchlight(f, pool_size=pool_size)

# Only save the data if this is the first core
if rank == 0: 

    # Convert the output into what can be used
    sl_result = sl_result.astype('double')
    sl_result[np.isnan(sl_result)] = 0  # If there are nans we want this

    # Save the volume
    sl_nii = nib.Nifti1Image(sl_result, affine_mat)
    hdr = sl_nii.header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    nib.save(sl_nii, output_name)  # Save
    
    print('Finished searchlight')
