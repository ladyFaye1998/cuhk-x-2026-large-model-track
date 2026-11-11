"""Independent auxiliary supervision from partially observed training actions.

Preserves original task-specific targets. Single/multi questions supply direct
positive/negative choices; selected combinations and sequence options supply
positive evidence only. Contradictory direct evidence is masked in this
auxiliary loss, while the original question-specific labels remain unchanged.
"""
import numpy as np

def build_partial_targets(original,rows,paths,emotion_terms=None):
    target=original.build_targets(rows,paths)
    pindex={path:i for i,path in enumerate(paths)}
    positive=np.zeros((len(paths),40),dtype=bool)
    negative=np.zeros_like(positive)
    for row in rows:
        index=pindex[row['path']];category=row['category']
        if category in ('single','multi'):
            for letter in original.LETTERS:
                action=original.parse_actions(row[letter])
                assert len(action)==1
                (positive if letter in row['answer'] else negative)[index,action[0]]=True
        elif category=='combination':
            assert len(row['answer'])==1
            positive[index,original.parse_actions(row[row['answer']])]=True
        elif category=='sequence':
            for letter in original.LETTERS:positive[index,original.parse_actions(row[letter])]=True
    conflict=positive&negative
    target['presence']=positive.astype(np.float32)
    target['known']=positive^negative
    if emotion_terms is not None:align_emotion_terms(target,emotion_terms)
    receipt=dict(paths=len(paths),positive_pairs=int(positive.sum()),negative_pairs=int(negative.sum()),
                 contradictory_pairs_masked=int(conflict.sum()),known_pairs=int(target['known'].sum()),
                 original_question_specific_labels_preserved=True,combination_unmentioned_actions_marked_negative=False,
                 sequence_positive_presence_used=True)
    return target,receipt

def align_emotion_terms(target,emotion_terms):
    """Align class indexes to the fixed vocabulary of training question options."""
    assert len(set(emotion_terms))==len(emotion_terms)
    index={term:i for i,term in enumerate(emotion_terms)}
    remap=np.asarray([index[term] for term in target['emotion_terms']],dtype=np.int64)
    target['emotion_opts']=remap[target['emotion_opts']]
    target['emotion_terms']=list(emotion_terms)
    return target
