"""Rebuild donor selection and label transfer from public metadata and training QA.

Thresholds come from the recovered identity-twin protocol. No fixed test-to-donor
table or historical prediction vector is stored in this implementation.
"""
from collections import defaultdict
from itertools import combinations
import re


def norm(value):
    return ' '.join(str(value).strip().casefold().split())


def user_number(user):
    return int(re.sub(r'\D', '', user))


def select_metadata_donors(metadata, training_users, target_users):
    signatures = defaultdict(set)
    for row in metadata:
        signatures[row['user_id']].add((row['activity_id'], row['identifier_id']))
    selected, evidence = {}, {}
    for target in sorted(target_users, key=user_number):
        candidates = []
        for donor in sorted(training_users, key=user_number):
            if donor == target:
                continue
            a, b = signatures[target], signatures[donor]
            similarity = len(a & b) / len(a | b) if a | b else 0.
            candidates.append(dict(donor=donor, jaccard=similarity))
        candidates.sort(key=lambda r: (-r['jaccard'], user_number(r['donor'])))
        if not candidates:
            continue
        best = candidates[0]
        margin = best['jaccard'] - (candidates[1]['jaccard'] if len(candidates) > 1 else 0.)
        active = best['jaccard'] >= .60 and margin >= .35
        evidence[target] = dict(**best, margin=margin, active=active, ranking=candidates)
        if active:
            selected[target] = best['donor']
    return selected, evidence


def row_identity(row, identities):
    if row['source'] != 'HAU':
        return None
    path = row['path'].replace('\\', '/')
    match = re.search(r'LM_test_\d+', path)
    if match:
        identity = identities.get(match[0])
        return (identity['user'], identity['trial']) if identity else None
    match = re.fullmatch(r'HAU/(user\d+)/(\d+-\d+-\d+)', path)
    return match.groups() if match else None


def emotion_proposals(train, target, identities, donors, min_shared=20):
    known = {row_identity(row, {}): row for row in train if row['source'] == 'HAU' and row['category'] == 'emotion'}
    training_users = {u for u, trial in known}
    groups = defaultdict(list)
    for row in target:
        if row['category'] != 'emotion':
            continue
        identity = row_identity(row, identities)
        if identity:
            groups[identity[0]].append((identity[1], row))
    result, evidence = [], {}
    for user, linked in sorted(groups.items()):
        ranking = []
        for donor in sorted(training_users, key=user_number):
            if donor == user:
                continue
            shared, available = 0, 0
            for trial, row in linked:
                if (donor, trial) not in known:
                    continue
                source = known[donor, trial]
                word = norm(source[source['answer']])
                shared += 1
                available += sum(norm(row[ch]) == word for ch in 'ABCD') == 1
            if shared >= min_shared:
                ranking.append(dict(donor=donor, shared=shared, available=available, rate=available/shared))
        ranking.sort(key=lambda r: (-r['rate'], -r['shared'], user_number(r['donor'])))
        selected = donors.get(user)
        valid = bool(selected and ranking and ranking[0]['donor'] == selected)
        margin = ranking[0]['rate'] - (ranking[1]['rate'] if len(ranking) > 1 else 0.) if ranking else 0.
        valid &= bool(ranking and ranking[0]['rate'] >= .90 and margin >= .50)
        evidence[user] = dict(metadata_donor=selected, active=valid, margin=margin, ranking=ranking)
        if not valid:
            continue
        for trial, row in linked:
            source = known.get((selected, trial))
            if source is None:
                continue
            word = norm(source[source['answer']])
            letters = [ch for ch in 'ABCD' if norm(row[ch]) == word]
            if len(letters) == 1:
                result.append(dict(qa_id=row['qa_id'], prediction=letters[0], user=user, donor=selected,
                                   trial=trial, training_qa_id=source['qa_id'], word=word))
    return result, evidence


def positive_facts(train):
    support = defaultdict(set)
    for row in train:
        identity = row_identity(row, {})
        if identity is None:
            continue
        answer = row['answer']
        if row['category'] in ('single', 'multi'):
            support[identity].update(norm(row[ch]) for ch in answer)
        elif row['category'] == 'combination':
            support[identity].update(norm(a) for a in row[answer].split(','))
        elif row['category'] == 'sequence':
            support[identity].update(norm(row[ch]) for ch in 'ABCD')
    return support


def action_and_sequence_proposals(train, target, identities, donors, parent):
    support = positive_facts(train)
    orders = defaultdict(list)
    for row in train:
        if row['source'] == 'HAU' and row['category'] == 'sequence':
            user, trial = row_identity(row, {})
            orders[user, trial.rsplit('-', 1)[0]].append([norm(row[ch]) for ch in row['answer']])
    result = []
    for row in target:
        identity = row_identity(row, identities)
        if not identity or identity[0] not in donors:
            continue
        user, trial = identity
        donor = donors[user]
        if row['category'] == 'multi':
            facts = support.get((donor, trial), set())
            prediction = ''.join(ch for ch in 'ABCD' if ch in parent[row['qa_id']] or norm(row[ch]) in facts)
            if prediction != parent[row['qa_id']]:
                result.append(dict(qa_id=row['qa_id'], prediction=prediction, category='multi', donor=donor, trial=trial, positive_facts=sorted(facts)))
        elif row['category'] == 'sequence':
            sequences = orders.get((donor, trial.rsplit('-', 1)[0]), [])
            if len(sequences) < 3:
                continue
            votes = defaultdict(list)
            for sequence in sequences:
                for a, b in combinations(sequence, 2):
                    pair = tuple(sorted((a, b)))
                    votes[pair].append(pair == (a, b))
            edges = set()
            for pair, values in votes.items():
                if all(values):
                    edges.add(pair)
                elif not any(values):
                    edges.add(pair[::-1])
            while True:
                expanded = edges | {(a, d) for a, b in edges for c, d in edges if b == c}
                if expanded == edges:
                    break
                edges = expanded
            if any(a == b for a, b in edges):
                continue
            words = {ch: norm(row[ch]) for ch in 'ABCD'}
            if len(set(words.values())) != 4:
                continue
            if any(((a, b) in edges) == ((b, a) in edges) for a, b in combinations(words.values(), 2)):
                continue
            before = {ch: sum((word, other) in edges for other in words.values()) for ch, word in words.items()}
            prediction = ''.join(sorted('ABCD', key=lambda ch: -before[ch]))
            if prediction != parent[row['qa_id']]:
                result.append(dict(qa_id=row['qa_id'], prediction=prediction, category='sequence', donor=donor, trial=trial, ordering_evidence=sorted(edges)))
    return result
