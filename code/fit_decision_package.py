"""Train the connected decision package and evaluate its five stages in order."""
from train_decisions import *
import pair_grouping as pair
import train_context_experts as context_expert

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();root=args.training
    rows=read(root/'training_qa.csv');archive=np.load(root/'features.npz',allow_pickle=False);paths=archive['paths'].tolist();features=archive['features']
    if {user(r) for r in rows}!=set(sum((list(f) for f in FOLDS),[])):raise ValueError('Unexpected training subjects')
    original={r['qa_id']:r for r in read(root/'original_oof.csv')};partial={r['qa_id']:r for r in read(root/'partial_oof.csv')}
    clips=training_clips(root);known_g,known_p=known_training_topology(paths,features)
    lock={'sources':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'test_inputs_read':False,'folds':FOLDS,'group_probability_threshold':.90,'gate':'Positive aggregate net and nonnegative in every fold; sequence also permits no observed changes because its unique-order certificate is independently defined.'}
    save(out/'policy_before_validation.json',lock);folds=[];manner_metrics=[]
    for f,held in enumerate(FOLDS):
        train=[r for r in rows if user(r) not in held];test=[r for r in rows if user(r) in held and r['source']=='HAU']
        ti=[i for i,p in enumerate(paths) if int(re.search(r'user(\d+)',p)[1]) not in held];vi=[i for i,p in enumerate(paths) if int(re.search(r'user(\d+)',p)[1]) in held]
        tp=[paths[i] for i in ti];vp=[paths[i] for i in vi];tx=features[ti];vx=features[vi]
        group_model=pair.fit(train,tp,tx);train_group=pair.predict(group_model,train,tp,tx);test_group=pair.predict(group_model,test,vp,vx)
        groups,paces,details=test_group;purity={'groups':len(details),'correct':sum(len({p.rsplit('-',1)[0] for p in d['members']})==1 for d in details)}
        lexicon=fit_manner(train);mx,mr=context_expert.matrix(train,tp,tx,lexicon,train_group);my=np.array([int(c==r['answer']) for r in mr for c in LETTERS])
        manner_model=context_expert.make().fit(mx,my);manner=context_expert.predict(manner_model,test,vp,vx,lexicon,test_group)
        mm=metric([r for r in test if r['category']=='emotion'],{q:r['prediction'] for q,r in partial.items()},manner);manner_metrics.append(mm)
        voter=fit_voter(prepared(train,known_g,known_p));votes=predict_voter(voter,prepared(test,groups,paces))
        graph=ClipGraphDecoder().fit(pd.DataFrame(train));graph_pred=dict(graph.predict(pd.DataFrame(test)).itertuples(index=False,name=None))
        base={r['qa_id']:partial[r['qa_id']]['prediction'] for r in test};fold_clips={p:c for p,c in clips.items() if p in groups}
        parent,detail=parent_predictions(test,base,partial,graph_pred,votes,fold_clips,groups)
        folds.append(dict(rows=test,base=base,parent=parent,clips=fold_clips,groups=groups,paces=paces,lexicon=lexicon,manner=manner,grouping=purity))
        print(json.dumps({'phase':'connected_training_fold','fold':f,'grouping':purity,'action_parent':metric([r for r in test if r['category'] in ACTION_CATEGORIES],base,parent),'manner':mm,'seconds':time.monotonic()-started}),flush=True)
    manner_enabled=sum(m['net'] for m in manner_metrics)>0 and all(m['net']>=0 for m in manner_metrics)
    for fold in folds:
        if manner_enabled:fold['parent'].update(fold['manner'])
        fold['current']=dict(fold['parent'])
    dispatch=[('constraint',lambda f:constraint_stage(f['rows'],f['current'])),('sequence',lambda f:sequence_stage(f['rows'],f['current'],f['clips'],f['groups'])),
        ('joint_consistency',lambda f:joint_stage(f['rows'],f['current'],f['clips'],f['groups'])),('presence_agreement',lambda f:agreement_stage(f['rows'],f['current'],original,partial)),
        ('manner_intersection',lambda f:manner_stage(f['rows'],f['current'],f['groups'],f['paces'],f['lexicon']))]
    gates={};records=[]
    for name,call in dispatch:
        outputs=[call(f) for f in folds];metrics=[metric(f['rows'],f['current'],value[0]) for f,value in zip(folds,outputs)]
        enabled=all(m['net']>=0 for m in metrics) and (sum(m['net'] for m in metrics)>0 or name=='sequence')
        gates[name]=enabled
        for fold,value in zip(folds,outputs):
            if enabled:fold['current']=value[0]
        records.append({'stage':name,'enabled':enabled,'folds':metrics})
        print(json.dumps(records[-1]),flush=True)
    group_model=pair.fit(rows,paths,features);grouping=pair.predict(group_model,rows,paths,features)
    lexicon=fit_manner(rows);mx,mr=context_expert.matrix(rows,paths,features,lexicon,grouping);my=np.array([int(c==r['answer']) for r in mr for c in LETTERS])
    models={'recording_groups':group_model,'manner_model':context_expert.make().fit(mx,my),'compact_graph':ClipGraphDecoder().fit(pd.DataFrame(rows)),'option_voter':fit_voter(prepared(rows,known_g,known_p))}
    for name,model in models.items():joblib.dump(model,out/(name+'.joblib'),compress=3)
    save(out/'manner_lexicon.json',lexicon);save(out/'gates.json',gates);save(out/'parent_gates.json',{'manner':manner_enabled})
    for f,fold in enumerate(folds):save(out/f'fold_{f}_predictions.json',{'parent':fold['parent'],'final':fold['current']})
    report={'folds':FOLDS,'grouping':[f['grouping'] for f in folds],'manner_metrics':manner_metrics,'manner_enabled':manner_enabled,'parent_metrics':[metric(f['rows'],f['base'],f['parent']) for f in folds],'final_metrics':[metric(f['rows'],f['parent'],f['current']) for f in folds],'stages':records,'sources':lock['sources'],'test_or_selected_predictions_read':False,'training_feature_cache_used':True,'training_clip_oof_proxy_used':True,'seconds':time.monotonic()-started}
    save(out/'validation.json',report);print(json.dumps({'complete':True,'seconds':report['seconds'],'gates':gates}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=2):run(p.parse_args())
