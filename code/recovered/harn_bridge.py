"""Bind aligned clip action classes to HARn options using training QA only."""
from collections import Counter, defaultdict
import re

def norm(value):
    return ' '.join(str(value).casefold().strip().split())

def fit_bindings(rows):
    bindings=defaultdict(Counter)
    for row in rows:
        if row['source']!='HARn' or row['category'] not in ('single','object_interaction'):
            continue
        match=re.match(r'HARn/(\d+)_',row['path'].replace('\\','/'))
        if match:
            bindings[int(match[1]),row['category']][norm(row[row['answer']])]+=1
    return dict(bindings)

def predict(rows, unit_classes, bindings):
    output=[]
    for row in rows:
        if row['source']!='HARn':continue
        match=re.search(r'LM_test_\d+',row['path'])
        unit=match[0] if match else row['path']
        classes=unit_classes.get(unit,[])
        if not classes or len(set(classes))!=1:continue
        action_class=classes[0]
        words=bindings.get((action_class,row['category']),{})
        candidates=[ch for ch in 'ABCD' if norm(row.get(ch,'')) in words]
        if len(candidates)!=1:continue
        ch=candidates[0]
        output.append(dict(qa_id=row['qa_id'],prediction=ch,unit=unit,action_class=action_class,
            category=row['category'],training_word_support=words[norm(row[ch])],binding='unique_training_supported_option_for_aligned_clip_class'))
    return output
