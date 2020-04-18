"""
Run a functional searchlight. Choose "rsa" or "classify" as the kernel to pick the analysis you want to run
and choose a subject you want to run on. This is the same as anatomical searchlight except that it uses the data
that's been converted to functional space. 
"""
from cfg import *

sub=int(sys.argv[1])
kernel=sys.argv[2] 
bcvarname=sys.argv[3]

if sub<0 or sub>=16:
    print("Invalid subject number")
    sys.exit(1)
sub_number=sub 
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



mask=nib.load("3mm_mask.nii.gz").get_data().astype('bool')
data=nib.load(test_d+sub+"_test.nii.gz").get_data()
data=data[mask]
space=np.load(func_d+"space_"+str(sub_number)+".npy")

def rsa(bolddata,bcvar):
    bolddata_sl=bolddata.T
    alexnet=bcvar
    human=np.corrcoef(bolddata_sl[:,:])
    vec=human[np.triu(np.ones(human.shape),k=10).astype('bool')] 
    return pearsonr(vec,alexnet)[0]

def decode(bolddata,bcvar): 
    bolddata_sl=bolddata.T
    model=Ridge() 
    y=bcvar
    split_idx=530

    train_X=bolddata_sl[:split_idx,:]
    train_y=y[:split_idx,:] 

    test_X=bolddata_sl[split_idx:,:]
    test_y=y[split_idx:,:]

    model.fit(train_X,train_y)
    pred_y=model.predict(test_X)
    
    #print(pred_y.shape,test_y.shape)
    true_chunks=np.asarray([np.hstack(subarray) for subarray in np.array_split(test_y,25)])
    pred_chunks=np.asarray([np.hstack(subarray) for subarray in np.array_split(pred_y,25)])
    corr_mtx=compute_correlation(pred_chunks,true_chunks)

    max_idx =  np.argmax(corr_mtx, axis=1)
    ans=sum(max_idx == range(25)) / 25.0 
    #print(ans) 
    return ans 

def process(i): 
    #if len(space[i])<=50 or len(space[i])>=343:  
    #   return float('NaN')
    #else:
    if space[i,0]<0: 
        return np.nan  
    else:
        #n_neighbors=sizes[i].astype('int')
        #idx=space[i].astype('int')[:n_neighbors]
        idx=space[i].astype('int')
        if kernel=='sensory_rsa':
            return rsa(data[idx,:],bcvar)
        elif kernel=='semantic_decoding':  
            return decode(data[idx,:],bcvar) 


#You can easily parallelize this computation by splitting up different r1,r2 values. 
r1=0
r2=data.shape[0]
inputs=list(range(r1,r2))
results=np.asarray([process(i) for i in inputs])  
out_name=results_dir+sub+"_functional_SL_"+kernel+".npy"
np.save(results,out_name)







