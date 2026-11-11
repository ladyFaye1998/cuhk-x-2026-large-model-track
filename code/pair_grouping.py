"""Learn recording-family similarity from training videos and public questions."""
from collections import defaultdict
from itertools import combinations
import re
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

def words(value):return {' '.join(s.casefold().split()) for s in str(value).split(',') if s.strip()}
def path_of(r):
    parts=r['path'].replace('\\','/').split('/')
    return '/'.join(parts[:-2]) if len(parts)>=4 and parts[-2].casefold() in ('depth','depth_color','ir','thermal','radar','imu','skeleton') else '/'.join(parts)
def pair_matrix(rows,paths,features):
    index={p:i for i,p in enumerate(paths)};vocab=[defaultdict(set) for p in paths]
    for r in rows:
        p=path_of(r)
        if p in index:
            for c in 'ABCD':vocab[index[p]][r['category']].update(words(r[c]))
    ir=features[:,:,3594:4362].astype(np.float32)
    visual=np.concatenate([ir.mean(1),ir.std(1),ir.max(1)],1);visual/=np.linalg.norm(visual,axis=1,keepdims=True).clip(1e-9)
    cos=visual@visual.T;duration=features[:,:,-1].astype(np.float32).mean(1)
    pairs=list(combinations(range(len(paths)),2));matrix=[]
    for i,j in pairs:
        values=[float(cos[i,j]),abs(float(duration[i]-duration[j]))]
        for cat in ('emotion','sequence','single','multi','combination'):
            a=vocab[i][cat];b=vocab[j][cat];common=len(a&b);union=len(a|b)
            values.extend([common,common/max(union,1),min(len(a),len(b)),max(len(a),len(b))])
        a=set.union(*vocab[i].values()) if vocab[i] else set();b=set.union(*vocab[j].values()) if vocab[j] else set()
        values.extend([len(a&b),len(a&b)/max(len(a|b),1)])
        matrix.append(values)
    return np.asarray(matrix,np.float32),pairs,duration
def fit(rows,paths,features):
    x,pairs,_=pair_matrix(rows,paths,features)
    y=np.array([paths[i].rsplit('-',1)[0]==paths[j].rsplit('-',1)[0] for i,j in pairs],np.int8)
    rng=np.random.default_rng(20260822)
    keep=(y==1)|(x[:,0]>.95)|(x[:,2]>=2)|(rng.random(len(y))<.025)
    model=HistGradientBoostingClassifier(max_iter=160,learning_rate=.1,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=2.,class_weight='balanced',random_state=20260822).fit(x[keep],y[keep])
    return model
def predict(model,rows,paths,features):
    x,pairs,durations=pair_matrix(rows,paths,features);prob=model.predict_proba(x)[:,list(model.classes_).index(1)]
    score=np.full((len(paths),len(paths)),-1.,np.float64)
    for (i,j),p in zip(pairs,prob):score[i,j]=score[j,i]=p
    top=np.argsort(-score,axis=1,kind='stable')[:,:2];seen=set();details=[];groups={p:p for p in paths};paces={}
    for i in range(len(paths)):
        candidate=tuple(sorted((i,*map(int,top[i]))))
        if len(set(candidate))!=3 or candidate in seen:continue
        seen.add(candidate)
        if not all(set(top[j])==set(candidate)-{j} for j in candidate):continue
        minimum=min(score[a,b] for a,b in combinations(candidate,2))
        if minimum<.90:continue
        # No identifier contributes to a pair score or a pace rank.
        members=[paths[j] for j in candidate];key='|'.join(sorted(members))
        ranked=sorted(candidate,key=lambda j:-durations[j]);distinct=len(set(float(durations[j]) for j in candidate))==3
        for rank,j in enumerate(ranked,1):
            groups[paths[j]]=key
            if distinct:paces[paths[j]]=rank
        details.append({'members':members,'minimum_pair_probability':float(minimum),'duration_order':[paths[j] for j in ranked] if distinct else None})
    return groups,paces,details
