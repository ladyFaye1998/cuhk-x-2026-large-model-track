"""Train sequence-prior and raw-motion manner experts using held subjects."""
from pathlib import Path
from collections import defaultdict
import argparse,json,re,time
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from train_decisions import read,save,sha,FOLDS,user,metric
from decision_stages import group_recordings,fit_manner,norm,unit
from recovered.compact_graph import ClipGraphDecoder

def context(rows,paths,features,grouping=None):
    groups,paces,details=grouping if grouping is not None else group_recordings(paths,features)
    index={p:i for i,p in enumerate(paths)};selected=defaultdict(list)
    for r in rows:
        if r['source']=='HAU' and r['category']=='emotion':selected[groups.get(unit(r),unit(r))].append(r)
    shared={g:set.intersection(*[{norm(r[c]) for c in 'ABCD'} for r in rs]) if len(rs)>1 else set() for g,rs in selected.items()}
    return groups,paces,index,shared

def matrix(rows,paths,features,lexicon,grouping=None):
    groups,paces,index,shared=context(rows,paths,features,grouping)
    result=[];selected=[]
    for r in rows:
        if r['source']!='HAU' or r['category']!='emotion':continue
        p=unit(r);f=features[index[p]].astype(np.float32);pace=paces.get(p,0)
        # Raw per-window IR/depth motion, brightness and duration statistics.
        raw=f[:,-12:];summary=np.concatenate([raw.mean(0),raw.std(0),raw.min(0),raw.max(0)])
        words=[norm(r[c]) for c in 'ABCD'];means=[lexicon.get(w,{'pace_mean':2.})['pace_mean'] for w in words]
        vocabulary=shared[groups.get(p,p)]
        for j,word in enumerate(words):
            expected=float(means[j]);support=lexicon.get(word,{}).get('count',0)
            row=np.concatenate([summary,np.array([pace,expected,np.log1p(support),abs(pace-expected) if pace else 0.,float(word in vocabulary),len(vocabulary),min(means),max(means),expected-min(means),max(means)-expected,sum(v<expected for v in means)],np.float32)])
            result.append(row)
        selected.append(r)
    return np.asarray(result,np.float32),selected

def make():return HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,random_state=20260822)

def predict(model,rows,paths,features,lexicon,grouping=None):
    x,rs=matrix(rows,paths,features,lexicon,grouping);prob=model.predict_proba(x)[:,list(model.classes_).index(1)].reshape(-1,4)
    return {r['qa_id']:'ABCD'[int(p.argmax())] for r,p in zip(rs,prob)}

def run(args):
    out=args.output;out.mkdir(exist_ok=False,parents=True);root=args.training;start=time.monotonic()
    rows=read(root/'training_qa.csv');data=np.load(root/'features.npz',allow_pickle=False);paths=data['paths'].tolist();features=data['features']
    parent={r['qa_id']:r['prediction'] for r in read(root/'partial_oof.csv')};records=[];predictions={}
    save(out/'policy_before_validation.json',{'source_sha256':sha(__file__),'test_inputs_read':False,'gates':'Positive net overall and nonnegative in every fold for each category.'})
    for fold,held in enumerate(FOLDS):
        train=[r for r in rows if user(r) not in held];test=[r for r in rows if user(r) in held and r['source']=='HAU']
        train_i=[i for i,p in enumerate(paths) if int(re.search(r'user(\d+)',p)[1]) not in held];test_i=[i for i,p in enumerate(paths) if int(re.search(r'user(\d+)',p)[1]) in held]
        lex=fit_manner(train);x,rs=matrix(train,[paths[i] for i in train_i],features[train_i],lex);y=np.array([int(c==r['answer']) for r in rs for c in 'ABCD'])
        model=make().fit(x,y);emotion=predict(model,test,[paths[i] for i in test_i],features[test_i],lex)
        graph=ClipGraphDecoder().fit(pd.DataFrame(train));sequence=dict(graph.predict(pd.DataFrame([r for r in test if r['category']=='sequence'])).itertuples(index=False,name=None))
        pred={**emotion,**sequence};predictions.update(pred)
        m={cat:metric([r for r in test if r['category']==cat],parent,pred) for cat in ('emotion','sequence')};records.append(m)
        print(json.dumps({'fold':fold,'metrics':m,'seconds':time.monotonic()-start}),flush=True)
    gates={cat:sum(m[cat]['net'] for m in records)>0 and all(m[cat]['net']>=0 for m in records) for cat in ('emotion','sequence')}
    lex=fit_manner(rows);x,rs=matrix(rows,paths,features,lex);y=np.array([int(c==r['answer']) for r in rs for c in 'ABCD'])
    model=make().fit(x,y);joblib.dump(model,out/'manner_model.joblib',compress=3)
    graph=ClipGraphDecoder().fit(pd.DataFrame(rows));joblib.dump(graph,out/'sequence_graph.joblib',compress=3)
    save(out/'manner_lexicon.json',lex);save(out/'gates.json',gates);save(out/'oof_predictions.json',predictions)
    save(out/'validation.json',{'folds':records,'gates':gates,'test_inputs_read':False,'source_sha256':sha(__file__),'seconds':time.monotonic()-start})
    print(json.dumps({'complete':True,'gates':gates}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args())
