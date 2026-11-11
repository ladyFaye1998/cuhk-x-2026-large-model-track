"""Fit the compact affine head to pseudo-labels generated during this run."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import dst_ridge_core_v2 as core

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fit_student(assets, output, sequences, counts, pseudo_labels):
    assets=Path(assets);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.manual_seed(core.SEED);torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    state=torch.load(assets/'native_state.pt',map_location='cpu',weights_only=True)
    if core.state_dict_sha256(state)!=core.EXPECTED_DST_STATE_SHA256:raise RuntimeError('Native state mismatch')
    model=core.build_dstformer_model(Path(__file__).parent,state)
    features,extraction=core.extract_features(model,sequences,counts,1024)
    scaler=StandardScaler();standardized=scaler.fit_transform(features.astype(np.float64))
    head=RidgeClassifier(alpha=1e-8,solver='svd');head.fit(standardized,pseudo_labels)
    if not np.array_equal(head.classes_,np.arange(40)):raise RuntimeError('Incomplete class coverage')
    fitted=head.predict(standardized).astype(np.int64)
    if not np.array_equal(fitted,pseudo_labels):raise RuntimeError('Student failed to reproduce its generated pseudo-targets')
    checkpoint=output/'skeleton_model.pt'
    audit=core.save_deployable_checkpoint(checkpoint,state,1024,scaler.mean_,scaler.scale_,head.coef_,head.intercept_,{
        'teacher_inputs_sha256':sha(assets/'teacher_inputs.npz'),
        'student_training_source_sha256':sha(Path(__file__)),
        'native_state_sha256':sha(assets/'native_state.pt'),
    })
    info={'fit_rows':len(pseudo_labels),'feature_dimension':1024,'student_pseudo_target_agreement':float((fitted==pseudo_labels).mean()),
          'training_seconds':time.perf_counter()-started,'checkpoint_bytes':checkpoint.stat().st_size,
          'checkpoint_sha256':sha(checkpoint),'feature_extraction':extraction,
          'method':'Frozen native encoder, StandardScaler and RidgeClassifier(alpha=1e-8, solver=svd) fitted to model-generated pseudo-labels.',
          'interpretation':'Pseudo-target agreement is a training fit check, not a ground-truth accuracy or held-subject result.'}
    (output/'student_fit_report.json').write_text(json.dumps(info,indent=2)+'\n')
    return checkpoint,info

def predict(checkpoint,sequences,counts):
    torch.set_num_threads(4);torch.manual_seed(core.SEED);torch.use_deterministic_algorithms(True)
    payload,audit=core.load_deployable_checkpoint(Path(checkpoint))
    model=core.build_dstformer_model(Path(__file__).parent,payload['dstformer_state'])
    features,extraction=core.extract_features(model,sequences,counts,int(payload['contract']['feature_dimension']))
    decision=core.linear_decision(features,payload['feature_mean'].numpy(),payload['feature_scale'].numpy(),payload['ridge_coefficient'].numpy(),payload['ridge_intercept'].numpy())
    return payload['class_order'].numpy()[decision.argmax(1)].astype(np.int64),extraction
