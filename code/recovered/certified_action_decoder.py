"""Independent same-video action decoder with a certificate of optimum agreement.

Uses the recovered fixed weights (8,8,5 after exact integer scaling), but never
arbitrarily selects an answer from tied optimum worlds. No recording families,
ordinal identities, test annotations, model APIs or final V2 answers are read.
The decoder operates only on current-run model evidence and public question text.
"""
import argparse, collections, csv, hashlib, itertools, json, pathlib, re, time
from z3 import And, Bool, If, Not, Optimize, Or, Solver, Sum, sat, unsat
WEIGHTS = {'single': 8, 'combination': 8, 'multi_choice': 5}
CATEGORIES = {'single', 'combination', 'multi', 'sequence'}

def options(row):
    result = {}
    for letter in 'ABCD':
        text = ' '.join(str(row.get(letter, '')).split()).casefold()
        if text:
            result[letter] = tuple((s.strip() for s in text.split(','))) if row['category'] == 'combination' else (text,)
    return result

def answer_condition(expressions, answer):
    return And([value if letter in answer else Not(value) for letter, value in expressions.items()])

def decode(questions, parent):
    forbidden = {'answer', 'prediction', 'target', 'label'}
    if any((forbidden & set(row) for row in questions)):
        raise ValueError('Answer-bearing question inputs forbidden')
    groups = collections.defaultdict(list)
    for row in questions:
        if row['source'] == 'HAU' and row['category'] in CATEGORIES:
            groups[row['path'].replace('\\', '/').rstrip('/')].append(row)
    result = []
    for path, group in sorted(groups.items()):
        opt = {row['qa_id']: options(row) for row in group}
        atoms = sorted({a for value in opt.values() for names in value.values() for a in names})
        variables = {a: Bool('a' + str(i)) for i, a in enumerate(atoms)}
        expressions = {q: {letter: And([variables[a] for a in names]) for letter, names in choices.items()} for q, choices in opt.items()}
        hard = []
        terms = []
        for row in group:
            q = row['qa_id']
            category = row['category']
            values = expressions[q]
            assert values, 'No answer options'
            if category in ('single', 'combination'):
                hard.append(Sum([If(v, 1, 0) for v in values.values()]) == 1)
            elif category == 'multi':
                hard.append(Or(list(values.values())))
            else:
                hard.append(And(list(values.values())))
            previous = parent.get(q)
            if previous is None:
                continue
            assert previous and set(previous) <= set(values), (q, 'Invalid parent choice')
            if category in ('single', 'combination'):
                assert len(previous) == 1
                terms.append(If(values[previous], WEIGHTS[category], 0))
            elif category == 'multi':
                terms.extend((If(value if letter in previous else Not(value), WEIGHTS['multi_choice'], 0) for letter, value in values.items()))
        score = Sum(terms) if terms else Sum([0])
        optimize = Optimize()
        optimize.add(*hard)
        objective = optimize.maximize(score)
        state = optimize.check()
        if state != sat:
            assert state == unsat
            result.extend((dict(qa_id=row['qa_id'], path=path, category=row['category'], parent=parent.get(row['qa_id']), prediction=parent.get(row['qa_id']), status='inconsistent_questions', changed=False) for row in group if row['category'] != 'sequence'))
            continue
        best = optimize.lower(objective).as_long()
        solver = Solver()
        solver.add(*hard, score == best)
        assert solver.check() == sat
        world = solver.model()
        for row in group:
            if row['category'] == 'sequence':
                continue
            q = row['qa_id']
            values = expressions[q]
            previous = parent.get(q)
            proposed = ''.join((letter for letter, value in values.items() if bool(world.evaluate(value, model_completion=True))))
            assert proposed and (row['category'] == 'multi' or len(proposed) == 1)
            solver.push()
            solver.add(Not(answer_condition(values, proposed)))
            alternate = solver.check()
            solver.pop()
            assert alternate in (sat, unsat)
            certified = alternate == unsat
            prediction = proposed if certified else previous
            margin = None
            old_feasible = None
            if certified and previous is not None and (prediction != previous):
                counterfactual = Optimize()
                counterfactual.add(*hard, answer_condition(values, previous))
                counter = counterfactual.maximize(score)
                old_state = counterfactual.check()
                old_feasible = old_state == sat
                if old_feasible:
                    margin = best - counterfactual.lower(counter).as_long()
                    assert margin > 0
                else:
                    assert old_state == unsat
            result.append(dict(qa_id=q, path=path, category=row['category'], parent=previous, prediction=prediction, optimum_answer=proposed if certified else None, optimum_score=best, status='same_answer_in_all_optima' if certified else 'ambiguous_optimum_abstain', changed=prediction != previous, old_answer_feasible=old_feasible, margin_integer=margin))
    return result
