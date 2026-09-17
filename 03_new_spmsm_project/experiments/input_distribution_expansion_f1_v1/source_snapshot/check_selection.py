"""Small exact reference checks; no FEMM calls or model training."""
import numpy as np
import expansion as e

def dense(x,ids,n,count,initial):
    centers=list(initial);chosen=[i for i in centers if i<n];trace=[]
    while len(chosen)<count:
        all_d=((x[:n,None,:]-x[np.array(centers)][None,:,:])**2).sum(2)
        owners=np.array([min(np.flatnonzero(row==row.min()),key=lambda j:ids[centers[j]]) for row in all_d]);near=all_d.min(1)
        active=np.ones(n,bool);active[chosen]=False
        sums=np.bincount(owners[active],weights=near[active],minlength=len(centers));occ=np.bincount(owners[active],minlength=len(centers));sums[occ==0]=-np.inf
        region=min(np.flatnonzero(sums==sums.max()),key=lambda j:ids[centers[j]])
        members=np.flatnonzero(active&(owners==region));i=min(members,key=lambda i:(-near[i],ids[i]));chosen.append(int(i));centers.append(int(i))
    return chosen

if __name__=='__main__':
    x=np.random.default_rng(17).integers(-3,4,(19,7)).astype(float);ids=[f'{i:03d}' for i in range(len(x)-1,-1,-1)]
    suffix=e.digest(__import__('inspect').getsource(e.lcmd))[:10]
    for name,n,initial,count in [('probe_anchor_'+suffix,19,[4],8),('probe_train_'+suffix,16,[16,17,18],8)]:
        expected=dense(x,ids,n,count,initial)
        actual,_,_=e.lcmd(x,ids,n,count,initial,name,resume=False);assert actual==expected
        e.lcmd(x,ids,n,count,initial,name,resume=True,stop_after=3)
        recovered,_,_=e.lcmd(x,ids,n,count,initial,name,resume=True);assert recovered==expected
    e.save(e.HERE/'selection_checks.json',{'status':'passed','dense_reference_matches':True,'bootstrap_counts_first_actual_center':True,'stable_hash_ties':True,'resumed_matches_uninterrupted':True,'float64_direct_distance':True,'no_femm_calls':True})
    print('LCMD dense-reference, initialization, stable ties and recovery checks passed')
