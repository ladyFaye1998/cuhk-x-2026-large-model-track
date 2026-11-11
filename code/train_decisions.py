"""Fit option experts and validate refinement policies on training subjects."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse,csv,hashlib,json,re,time
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from recovered import triplet_features as tf
from recovered.compact_graph import ClipGraphDecoder
from recovered.action_taxonomy import LABEL_TO_ACTION
from decision_stages import *

FOLDS=((1,2,3),(4,5,6),(7,8,9),(16,17,18),(19,20,21),(22,23,24))
RULES={'single':('et',),'multi':('logistic',),'combination':('et','rf','hist','logistic')}
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def user(r):return int(re.search(r'user(\d+)',r['path'])[1])
def estimator(name):
    if name=='et':return ExtraTreesClassifier(n_estimators=500,min_samples_leaf=3,max_features=.8,class_weight='balanced',random_state=20260822,n_jobs=2)
    if name=='rf':return RandomForestClassifier(n_estimators=600,min_samples_leaf=3,max_features=.8,class_weight='balanced_subsample',random_state=20260875,n_jobs=2)
    if name=='hist':return HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=1.,random_state=20260893)
    return make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,random_state=20260911))
def prepared(rows,groups,paces):
    return [dict(r,_triplet=groups[unit(r)],_pace=paces[unit(r)]) for r in rows if r['source']=='HAU' and r['category'] in ACTION_CATEGORIES|{'sequence'} and unit(r) in paces]
def matrix(rows,category,names=None):
    grouped=defaultdict(list)
    for r in rows:grouped[r['_triplet']].append(r)
    context={g:tf.build_context(rs) for g,rs in grouped.items()}
    selected=[r for r in rows if r['category']==category]
    if not selected:return None,names,selected
    x,names=tf.vectorize([tf.option_features(r,c,context[r['_triplet']]) for r in selected for c in LETTERS],names)
    return x,names,selected
def fit_voter(rows):
    models={}
    for category,kinds in RULES.items():
        x,names,rs=matrix(rows,category);y=np.array([int(c in r['answer']) for r in rs for c in LETTERS])
        models[category]={'names':names,'models':[estimator(k).fit(x,y) for k in kinds]}
    return models
def predict_voter(models,rows):
    result={}
    for cat,entry in models.items():
        x,_,rs=matrix(rows,cat,entry['names'])
        if not rs:continue
        probabilities=np.mean([m.predict_proba(x)[:,list(m.classes_).index(1)] for m in entry['models']],axis=0).reshape(-1,4)
        for r,p in zip(rs,probabilities):
            result[r['qa_id']]=(''.join(c for c,v in zip(LETTERS,p) if v>=.52) if cat=='multi' else '') or LETTERS[int(p.argmax())]
    return result
def known_training_topology(paths,features):
    durations=dict(zip(paths,features[:,:,-1].astype(np.float32).mean(1)));grouped=defaultdict(list)
    for p in paths:grouped[p.rsplit('-',1)[0]].append(p)
    groups={};paces={}
    for g,part in grouped.items():
        if len(part)!=3:continue
        for i,p in enumerate(sorted(part,key=lambda p:-durations[p]),1):groups[p]=g;paces[p]=i
    return groups,paces
def training_clips(root):
    safe={r['path']:{int(v) for v in r['official_positive_labels'].split(';') if v} for r in read(root/'clip_eligibility.csv')}
    data=np.load(root/'clip_oof.npz',allow_pickle=False);labels=data['labels'];predictions=data['probabilities'].argmax(1)
    clips=defaultdict(list)
    for r,label,pred in zip(read(root/'clip_metadata.csv'),labels,predictions,strict=True):
        p=f"HAU/user{r['user']}/{r['trial']}"
        if p in safe and int(label) in safe[p]:
            timestamp=datetime.fromisoformat(r['start_time']).timestamp()
            clips[p].append({'action':LABEL_TO_ACTION[int(pred)],'offset':timestamp,'small_key':r['group'],'cross_modal_confirmed':False,'quality':1.0})
    return dict(clips)
def metric(rows,previous,after):
    total=len(rows);wins=losses=old=new=0
    for r in rows:
        a=previous[r['qa_id']]==r['answer'];b=after[r['qa_id']]==r['answer'];old+=a;new+=b;wins+=b and not a;losses+=a and not b
    return {'rows':total,'before_correct':old,'after_correct':new,'wins':wins,'losses':losses,'net':wins-losses,'accuracy':new/max(total,1)}
def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=read(args.training/'training_qa.csv')
    if {user(r) for r in rows}!=set(sum((list(f) for f in FOLDS),[])):raise ValueError('Unexpected training subjects')
    source_lock={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    save(out/'policy_before_validation.json',{'sources':source_lock,'test_inputs_read':False,'folds':FOLDS,'gate':'Each stage requires positive net wins overall and nonnegative net in every held-out fold. Stages are considered once in the specified order.','parent':'Partial temporal; compact/voter agreement; four-action clip routing.'})
    data=np.load(args.training/'features.npz',allow_pickle=False);paths=data['paths'].tolist();features=data['features']
    original={r['qa_id']:r for r in read(args.training/'original_oof.csv')};partial={r['qa_id']:r for r in read(args.training/'partial_oof.csv')}
    ownclips=training_clips(args.training);known_g,known_p=known_training_topology(paths,features)
    folds=[]
    for f,held in enumerate(FOLDS):
        train=[r for r in rows if user(r) not in held];test=[r for r in rows if user(r) in held and r['source']=='HAU']
        ids=[i for i,p in enumerate(paths) if int(re.search(r'user(\d+)',p)[1]) in held]
        gp,pace,triples=group_recordings([paths[i] for i in ids],features[ids]);actual=lambda p:p.rsplit('-',1)[0]
        purity={'groups':len(triples),'correct':sum(len({actual(p) for p in t['members']})==1 for t in triples)}
        voter=fit_voter(prepared(train,known_g,known_p));votes=predict_voter(voter,prepared(test,gp,pace))
        graph_model=ClipGraphDecoder().fit(pd.DataFrame(train));graph=dict(graph_model.predict(pd.DataFrame(test)).itertuples(index=False,name=None))
        base={r['qa_id']:partial[r['qa_id']]['prediction'] for r in test}
        clip={p:v for p,v in ownclips.items() if p in gp}
        parent,detail=parent_predictions(test,base,partial,graph,votes,clip,gp)
        folds.append({'rows':test,'parent':parent,'current':dict(parent),'clips':clip,'groups':gp,'paces':pace,'lexicon':fit_manner(train),'grouping':purity,'parent_metric':metric(test,base,parent),'original':original,'partial':partial})
        save(out/f'fold_{f}_parent.json',{'grouping':purity,'metric':metric(test,base,parent),'predictions':parent,'details':detail})
        print(json.dumps({'phase':'training_fold','fold':f,'grouping':purity,'metric':metric(test,base,parent),'seconds':round(time.monotonic()-start,1)}),flush=True)
    gates={};stage_reports=[]
    for index,name in enumerate(['constraint','sequence','joint_consistency','presence_agreement','manner_intersection']):
        proposals=[];metrics=[]
        for fold in folds:
            _,stages=five_stages(fold['rows'],fold['current'],fold['clips'],fold['groups'],fold['paces'],fold['lexicon'],original,partial,{n:n==name for n in ['constraint','sequence','joint_consistency','presence_agreement','manner_intersection']})
            candidate=stages[index]['predictions'];proposals.append(candidate);metrics.append(metric(fold['rows'],fold['current'],candidate))
        enabled=sum(m['net'] for m in metrics)>0 and all(m['net']>=0 for m in metrics);gates[name]=enabled
        if enabled:
            for fold,candidate in zip(folds,proposals):fold['current']=candidate
        stage_reports.append({'stage':name,'enabled':enabled,'folds':metrics})
        print(json.dumps(stage_reports[-1]),flush=True)
    full=fit_voter(prepared(rows,known_g,known_p));joblib.dump(full,out/'option_voter.joblib',compress=3)
    graph=ClipGraphDecoder().fit(pd.DataFrame(rows));joblib.dump(graph,out/'compact_graph.joblib',compress=3)
    save(out/'manner_lexicon.json',fit_manner(rows));save(out/'gates.json',gates)
    report={'folds':FOLDS,'parent_metrics':[f['parent_metric'] for f in folds],'final_metrics':[metric(f['rows'],f['parent'],f['current']) for f in folds],'stages':stage_reports,'raw_grouping':[f['grouping'] for f in folds],'test_or_selected_predictions_read':False,'training_feature_cache_used':True,'training_only_clip_oof_used':True,'clip_validation_note':'The retained Small clip OOF teacher is used as a proxy for clip routing. This is not a complete rerun of every encoder and current Small checkpoint in each fold.','seconds':time.monotonic()-start,'sources':source_lock,'model_files':{p.name:sha(p) for p in out.glob('*.joblib')}}
    save(out/'validation.json',report);print(json.dumps({'phase':'training_complete','gates':gates,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args())
