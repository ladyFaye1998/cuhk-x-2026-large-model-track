"""Original MIL training loop and complete model-to-graph prediction path."""
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from . import masked_mil_core as core


def initialize_model(train, features):
    ontology = core.build_ontology(train, features['train_units'])
    aliases = core.build_fullfit_aliases(train)
    semantics, _ = core.build_emotion_semantics(train, ontology['emotion_vocab'])
    model = core.MaskedMILV2(sensor_width=core.EXPECTED_SENSOR_WIDTH,
        vision_width=core.EXPECTED_VISION_WIDTH, n_steps=core.EXPECTED_STEPS,
        n_actions=len(ontology['class_names']), motion_width=core.EXPECTED_MOTION_WIDTH,
        emotion_semantics=semantics, emotion_vocab_size=len(ontology['emotion_vocab']),
        object_vocab_size=len(ontology['object_vocab']))
    return model, ontology, aliases


def fit_model(model, targets, features, device=torch.device('cpu'), progress=None, stop_after=None, num_workers=0):
    """Use the recovered 56-epoch recipe; labels come from original training QA."""
    dataset = core.UnitDataset(features['train_units'], targets, features['unit_index'],
        features['source_ids'], features['sensor'], features['vision'], features['motion'], features['modality'])
    loader = DataLoader(dataset, batch_size=core.BATCH_SIZE, shuffle=True, num_workers=num_workers,
                        pin_memory=device.type == 'cuda', drop_last=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=core.EPOCHS, eta_min=1.5e-5)
    history = []
    epochs=core.EPOCHS if stop_after is None else min(core.EPOCHS,stop_after)
    for epoch in range(epochs):
        model.train()
        losses, pieces, counts = [], defaultdict(float), defaultdict(int)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, detail = core.batch_loss(model, batch, device, core.EXPECTED_STEPS)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss in epoch {epoch + 1}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Non-finite gradient in epoch {epoch + 1}')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in detail.items():
                pieces[name] += value
                counts[name] += 1
        scheduler.step()
        row = dict(epoch=epoch + 1, mean_loss=sum(losses) / len(losses), batches=len(losses),
                   learning_rate=float(optimizer.param_groups[0]['lr']))
        row.update({name: pieces[name] / counts[name] for name in pieces})
        history.append(row)
        if progress:
            progress(row)
    return pd.DataFrame(history)


def predict(model, train, test, targets, features, ontology, aliases):
    model = model.to('cpu').eval()
    core.NUM_WORKERS = 0
    evidence = core.infer_units(model, features['test_units'], targets, features['unit_index'],
        features['source_ids'], features['sensor'], features['vision'], features['motion'], features['modality'],
        torch.device('cpu'))
    raw = {str(row.qa_id): core.decode_row_v2(row, evidence[str(row.unit)], aliases, ontology['class_index'])
           for _, row in test.iterrows()}
    rule_audit, enabled = core.exact_rule_audit(train, {})
    graph_model = core.learn_graph_model(train, {})
    hard, graph = dict(raw), dict(raw)
    for _, group in test.groupby('unit', sort=False):
        rows = core.clip_rows(group)
        group_raw = {str(row['qa_id']): raw[str(row['qa_id'])] for row in rows.values()}
        group_hard, _ = core.apply_hard_stage(rows, group_raw, enabled, {})
        group_graph, _, _ = core.graph_decode(rows, raw, group_hard, graph_model, enabled, {})
        hard.update(group_hard)
        graph.update(group_graph)
    result = test[['qa_id', 'source', 'category', 'path']].copy()
    result['raw_prediction'] = result.qa_id.map(raw)
    result['hard_prediction'] = result.qa_id.map(hard)
    result['prediction'] = result.qa_id.map(graph)
    result['raw_option_probabilities'] = [json.dumps(core.model_option_probabilities(
        row, evidence[str(row.unit)], aliases, ontology['class_index']), sort_keys=True)
        for _, row in test.iterrows()]
    return result, pd.DataFrame(rule_audit), evidence
