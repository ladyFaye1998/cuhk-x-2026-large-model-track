"""Evidence-driven parent assembly and five deterministic refinement stages."""
from collections import Counter,defaultdict
from itertools import combinations,permutations
import re
import numpy as np
from recovered.clip_action_routing import action_answer,atoms
from recovered import certified_action_decoder as certified
from recovered.ir_grouping_core import recover_triplets

LETTERS='ABCD'
ACTION_CATEGORIES={'single','multi','combination'}

def norm(s):return ' '.join(str(s).casefold().split())

def unit(row):
    path=row['path'].replace('\\','/').rstrip('/')
    parts=path.split('/')
    if len(parts)>=4 and parts[-2].casefold() in {'depth','depth_color','ir','thermal','radar','skeleton','imu'}:return '/'.join(parts[:-2])
    return path

def public(rows):
    return [{k:v for k,v in r.items() if k not in {'answer','prediction','target','label'}} for r in rows]

def group_recordings(paths,features):
    # These are normalized SigLIP image embeddings from raw temporal windows.
    image=features[:,:,3594:4362].astype(np.float32)
    embeddings=np.concatenate([image.mean(1),image.std(1),image.max(1)],axis=1)
    triples=recover_triplets(paths,embeddings,list(range(len(paths))),False)
    groups={p:p for p in paths};paces={};details=[]
    durations=dict(zip(paths,features[:,:,-1].astype(np.float32).mean(1)))
    for triple in triples:
        members=triple.members
        ordered=sorted(members,key=lambda p:-float(durations[p]))
        # Equal measured durations do not establish a pace order.
        distinct=len({float(durations[p]) for p in members})==3
        key=min(triple.member_keys)
        for i,p in enumerate(ordered,1):
            groups[p]=key
            if distinct:paces[p]=i
        details.append({'members':list(members),'minimum_cosine':triple.minimum_cosine,'reciprocal_margin':triple.reciprocal_margin,'duration_order':ordered if distinct else None})
    return groups,paces,details

def group_clips(clips,groups):
    evidence=defaultdict(set)
    for p,values in clips.items():evidence[groups.get(p,p)].update(norm(c['action']) for c in values)
    return {p:evidence[groups.get(p,p)] for p in groups}

def parent_predictions(rows,base,temporal,graph,voter,clips,groups):
    result=dict(base);details=[];evidence=group_clips(clips,groups)
    for r in rows:
        q=r['qa_id'];p=unit(r);old=result[q];reason='raw_mil'
        if r['source']=='HAU' and q in temporal:
            result[q]=temporal[q]['prediction'];reason='partial_temporal'
            if r['category'] in ACTION_CATEGORIES:
                if voter.get(q)==graph.get(q) and q in voter:result[q]=voter[q];reason='trained_option_models_agree'
                result[q],routing=action_answer(r,evidence.get(p,set()),result[q],minimum_actions=4)
                reason+=';'+routing['rule']
        details.append({'qa_id':q,'prediction':result[q],'reason':reason,'evidence_actions':sorted(evidence.get(p,set()))})
    return result,details

def constraint_stage(rows,parent):
    clean=public(rows)
    for r in clean:r['path']=unit(r)
    detail=certified.decode(clean,parent);out=dict(parent)
    for r in detail:out[r['qa_id']]=r['prediction']
    return out,detail

def sequence_stage(rows,parent,clips,groups):
    edges=defaultdict(set);own_edges=defaultdict(set)
    for p,values in clips.items():
        occurrences=defaultdict(list)
        for c in values:occurrences[norm(c['action'])].append(float(c['offset']))
        for a,b in permutations(occurrences,2):
            if max(occurrences[a])<min(occurrences[b]):
                edges[groups.get(p,p)].add((a,b));own_edges[p].add((a,b))
    out=dict(parent);detail=[]
    for r in rows:
        if r['source']!='HAU' or r['category']!='sequence':continue
        p=unit(r);pairs=own_edges[p];names={c:norm(r[c]) for c in LETTERS}
        reachable=set(pairs)
        while True:
            new=reachable|{(a,d) for a,b in reachable for c,d in reachable if b==c}
            if new==reachable:break
            reachable=new
        relevant={(a,b) for a,b in reachable if a in names.values() and b in names.values()}
        winners=[]
        if not any(a==b for a,b in reachable):
            for order in permutations(LETTERS):
                rank={names[c]:i for i,c in enumerate(order)}
                if all(rank[a]<rank[b] for a,b in relevant):winners.append(''.join(order))
        if len(winners)==1:out[r['qa_id']]=winners[0]
        detail.append({'qa_id':r['qa_id'],'prediction':out[r['qa_id']],'feasible_orders':len(winners),'order_edges':sorted(relevant),'status':'unique_order' if len(winners)==1 else 'ambiguous_or_conflicting_order_abstain'})
    return out,detail

def joint_stage(rows,parent,clips,groups):
    # Pace variants contribute action presence; ordering is handled separately.
    clean=public(rows)
    for r in clean:r['path']=groups.get(unit(r),unit(r))
    detail=certified.decode(clean,parent);out=dict(parent)
    for r in detail:out[r['qa_id']]=r['prediction']
    return out,detail

def agreement_stage(rows,parent,first,second):
    out=dict(parent);detail=[];by_path=defaultdict(list)
    for r in rows:by_path[unit(r)].append(r)
    for p,group in sorted(by_path.items()):
        proposed=dict(out);candidates=[]
        for r in group:
            q=r['qa_id'];a=first.get(q);b=second.get(q)
            if r['source']!='HAU' or r['category']!='multi' or not a or not b:continue
            if a['prediction']!=b['prediction'] or a['prediction']==out[q]:continue
            if min(float(a['confidence']),float(b['confidence']))<.60 or min(float(a['margin']),float(b['margin']))<.20:continue
            added=''.join(c for c in LETTERS if c in out[q] or c in a['prediction'])
            if added==out[q]:continue
            proposed[q]=added;candidates.append(q)
        if not candidates:continue
        checked,certificates=constraint_stage(group,proposed)
        # The agreed answers must survive the same-video constraints together.
        accepted=all(checked[q]==proposed[q] for q in candidates)
        if accepted:
            for q in candidates:out[q]=proposed[q]
        detail.append({'unit':p,'questions':candidates,'accepted':accepted,'proposed':{q:proposed[q] for q in candidates},'certificates':certificates})
    return out,detail

def fit_manner(rows):
    values=defaultdict(list)
    for r in rows:
        if r['source']=='HAU' and r['category']=='emotion':
            match=re.search(r'-([123])$',r['path'])
            if match:values[norm(r[r['answer']])].append(int(match[1]))
    return {w:{'pace_mean':float(np.mean(v)),'count':len(v)} for w,v in values.items()}

def manner_stage(rows,parent,groups,paces,lexicon):
    grouped=defaultdict(list)
    for r in rows:
        if r['source']=='HAU' and r['category']=='emotion':grouped[groups.get(unit(r),unit(r))].append(r)
    out=dict(parent);detail=[]
    for key,part in sorted(grouped.items()):
        if len({unit(r) for r in part})<2:continue
        shared=set.intersection(*[{norm(r[c]) for c in LETTERS} for r in part])
        if len(shared)!=3 or not all(w in lexicon for w in shared):continue
        scores={w:float(lexicon[w]['pace_mean']) for w in shared}
        if len(set(scores.values()))!=3:continue
        order=sorted(shared,key=scores.get)
        for r in part:
            q=r['qa_id'];pace=paces.get(unit(r));old=parent[q]
            if pace not in (1,2,3) or norm(r[old]) in shared:continue
            used={norm(s[parent[s['qa_id']]]) for s in part if s['qa_id']!=q}
            word=order[pace-1]
            if word in used:continue
            winners=[c for c in LETTERS if norm(r[c])==word]
            if len(winners)!=1:continue
            out[q]=winners[0]
            detail.append({'qa_id':q,'prediction':out[q],'pace_rank':pace,'shared_vocabulary':order,'training_pace_means':scores,'sibling_selected_words':sorted(used)})
    return out,detail

def five_stages(rows,parent,clips,groups,paces,lexicon,original,partial,gates=None):
    methods=[('constraint',lambda p:constraint_stage(rows,p)),('sequence',lambda p:sequence_stage(rows,p,clips,groups)),
        ('joint_consistency',lambda p:joint_stage(rows,p,clips,groups)),('presence_agreement',lambda p:agreement_stage(rows,p,original,partial)),
        ('manner_intersection',lambda p:manner_stage(rows,p,groups,paces,lexicon))]
    result=dict(parent);records=[]
    for name,call in methods:
        candidate,detail=call(result);enabled=gates is None or gates.get(name,False)
        changes=[{'qa_id':q,'before':result[q],'after':candidate[q]} for q in sorted(result) if candidate[q]!=result[q]]
        records.append({'stage':name,'enabled':enabled,'proposed_changes':changes,'details':detail,'predictions':candidate if enabled else dict(result)})
        if enabled:result=candidate
    return result,records
