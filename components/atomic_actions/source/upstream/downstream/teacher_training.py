"""Fit downstream teachers from declared upstream probabilities and train labels."""
from pathlib import Path
import json
import hashlib
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import consensus_ops as con
import prune_ops as pruning
import batch_ops as batch
import presence_ops as presence
import residual_ops as residual
import geometry_ops as geometry
import geometry_deploy as geometry_deployment
import ranked_ops as ranked

FOLDS=((1,2,3),(4,5,6),(7,8,9),(16,17,18),(19,20,21),(22,23,24))

def fit_teachers(assets, output):
    assets=Path(assets);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with np.load(assets/'teacher_inputs.npz',allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    labels=data['labels'].astype(np.int64)
    users,batches,folds=residual.load_meta(assets/'train_sensor_metadata.csv')
    held_batches=geometry_deployment.test_batches(assets/'test_sensor_metadata.csv')
    assert np.array_equal(users,data['users']) and np.array_equal(folds,data['folds'])
    assert len(labels)==2931 and np.array_equal(np.unique(labels),np.arange(40))
    for rows in batches:assert len(np.unique(labels[rows]))==len(rows)
    views={}
    parents={}
    for split,bs in [('train',batches),('test',held_batches)]:
        anchor=con.normalize(data[split+'_anchor'])
        teachers=[con.normalize(data[split+'_'+name]) for name in ['sensor','sequence','kinematic','fusion']]
        y,s,c=con.consensus(teachers)
        selected=con.batch_postprocess(con.correction(anchor,y,s,c,4,.75,.5),bs)
        parents[split],_,_=pruning.prune(anchor,teachers,selected.argmax(1))
        views[split]=[con.normalize(selected),anchor,*teachers]
        # Preserve the original boundary: batch design uses the upstream views;
        # alignment uses the consensus output before its subsequent normalization.
        views[split+'_presence']=[anchor,*teachers]
        views[split+'_aligned']=batch.align_probabilities(selected,parents[split])
    design=presence.batch_design(views['train_presence'],batches)
    target=presence.batch_targets(labels,batches)
    batch_users=np.array([users[rows[0]] for rows in batches])
    presence_scores=np.zeros((len(batches),40))
    fitted=[]
    for held in FOLDS:
        mask=np.isin(batch_users,held)
        model=LogisticRegression(C=.03,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=20260823)
        model.fit(design[~mask].reshape(-1,design.shape[2]),target[~mask].reshape(-1))
        presence_scores[mask]=model.predict_proba(design[mask].reshape(-1,design.shape[2]))[:,1].reshape(mask.sum(),40)
        fitted.append(model)
    model=LogisticRegression(C=.03,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=20260823)
    model.fit(design.reshape(-1,design.shape[2]),target.reshape(-1))
    fitted.append(model);joblib.dump(fitted,output/'presence_models.joblib')
    held_design=presence.batch_design(views['test_presence'],held_batches)
    held_presence=model.predict_proba(held_design.reshape(-1,held_design.shape[2]))[:,1].reshape(len(held_batches),40)
    report=[]
    for split,bs,scores in [('train',batches,presence_scores),('test',held_batches,held_presence)]:
        decoded,details=presence.decode(views[split+'_aligned'],parents[split],bs,scores,.25)
        parents[split]=presence.gate(parents[split],decoded,bs,details,1,0.)
    def record(name,diagnostics=None):
        row={'stage':name,'downstream_fold_correct':int((parents['train']==labels).sum()),'training_rows':len(labels)}
        if diagnostics:row['pair_table']=diagnostics
        report.append(row);print(json.dumps(row),flush=True)
    record('batch_presence')
    def logistic_stage(name,c):
        x,y,rows,candidates,numeric,diag=residual.build_pair_table(views['train'],labels,parents['train'],batches,1)
        xt,_,rt,ct,nt,_=residual.build_pair_table(views['test'],np.full(405,-1),parents['test'],held_batches,1)
        assert numeric==nt
        scores=residual.crossfit_scores(x,y,rows,folds,'logistic',{'C':c},numeric)
        model=residual.logistic_model(c,numeric,x.shape[1]);model.fit(x,y)
        test_scores=model.predict_proba(xt)[:,1]
        parents['train']=residual.decode(parents['train'],rows,candidates,scores,batches,.99,0.,1)
        parents['test']=residual.decode(parents['test'],rt,ct,test_scores,held_batches,.99,0.,1)
        joblib.dump(model,output/(name+'_model.joblib'))
        record(name,diag)
    logistic_stage('stacked_residual',3.)
    geo_train,_=geometry.cross_subject_geometry_views(data['train_features'].astype(np.float64),labels,folds)
    geo_test=geometry_deployment.fullfit_geometry_views(geometry,data['train_features'].astype(np.float64),data['test_features'].astype(np.float64),labels)
    views['train']+=geo_train;views['test']+=geo_test
    logistic_stage('feature_geometry',1.)
    for split in ['train','test']:
        skate=data[split+'_skate'] if split=='train' else con.normalize(data[split+'_skate'])
        egcn=data[split+'_egcn'] if split=='train' else con.normalize(data[split+'_egcn'])
        views[split]+=[skate,egcn,con.normalize(data[split+'_anchor']),ranked.normalize_logits(data[split+'_raw_logits'])]
    x,y,rows,candidates,numeric,diag=residual.build_pair_table(views['train'],labels,parents['train'],batches,1)
    xt,_,rt,ct,nt,_=residual.build_pair_table(views['test'],np.full(405,-1),parents['test'],held_batches,1)
    assert numeric==nt
    scores=residual.crossfit_scores(x,y,rows,folds,'hist',{'learning_rate':.03,'max_leaf_nodes':15},numeric)
    # The recovered protocol fixes 2 training interventions, scaled to 1 test
    # intervention. There is no reference-output score or parameter search here.
    parents['train'],_=ranked.ranked_decode(parents['train'],rows,candidates,scores,batches,2)
    model=residual.hist_model(.03,15)
    positive=int(y.sum());negative=len(y)-positive
    weights=np.where(y==1,len(y)/(2*positive),len(y)/(2*negative))
    model.fit(x,y,sample_weight=weights)
    parents['test'],_=ranked.ranked_decode(parents['test'],rt,ct,model.predict_proba(xt)[:,1],held_batches,1)
    joblib.dump(model,output/'adaptive_residual_model.joblib');record('adaptive_residual',diag)
    pseudo=parents['test'].astype(np.int64)
    np.save(output/'generated_pseudo_labels.npy',pseudo)
    info={'stages':report,'cpu_seconds':time.perf_counter()-started,
          'scope':'This function fits downstream models using newly regenerated and verified upstream stage arrays. Upstream fitting and inference are recorded in the input_provenance stage manifests.',
          'validation_limit':'Native supervised pretraining included all official training sequences. Downstream folds do not establish unseen-subject validation of the full pipeline.',
          'upstream_fitting_performed_by_this_function':False}
    (output/'teacher_fit_report.json').write_text(json.dumps(info,indent=2)+'\n')
    return pseudo,info
