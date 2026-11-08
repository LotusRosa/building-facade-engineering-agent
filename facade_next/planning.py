"""Model-free reproducible exposure planning/mining."""
import math
import random


def initial_pos_weight(rows):
    result = []
    for j in range(3):
        positive = sum(r['labels'][j]==1 for r in rows)
        negative = sum(r['labels'][j]==0 for r in rows)
        if not positive or not negative:
            raise ValueError('Initial Train needs positive and negative labels per class')
        result.append(float(min(20.,max(1.,negative/positive))))
    return result


def cycle(values,count,rng):
    if not values and count:
        raise ValueError('Empty sampling pool')
    result = []
    while len(result) < count:
        block = list(values)
        rng.shuffle(block)
        result.extend(block[:count-len(result)])
    return result


def batches(target,history,seed,epoch,steps,weights=None):
    tr = random.Random(seed*1000003+epoch*1009+17)
    hr = random.Random(seed*1000003+epoch*1009+29)
    sr = random.Random(seed*1000003+epoch*1009+43)
    count = steps*8
    if weights is None:
        # Retain original 300+4 schedule for the default 304 draw preset.
        targets = cycle(target,300,tr)+cycle(target,4,tr) if count==304 else cycle(target,count,tr)
    else:
        targets = tr.choices(target,weights=[weights[x] for x in target],k=count)
    old = cycle(history,count,hr)
    result = []
    for i in range(steps):
        batch = targets[i*8:i*8+8]+old[i*8:i*8+8]
        sr.shuffle(batch)
        result.append(batch)
    return result


def mining(rows,probs,thresholds,pos_weight,method):
    if method == 'hnm':
        return [i for i,(r,p) in enumerate(zip(rows,probs)) if any(y==0 and p[j]>=thresholds[j] for j,y in enumerate(r['labels']))],None
    weights = {}
    for i,(r,p) in enumerate(zip(rows,probs)):
        terms = []
        for j,y in enumerate(r['labels']):
            if y is not None:
                q = min(1-1e-7,max(1e-7,p[j]))
                terms.append(-(pos_weight[j]*y*math.log(q)+(1-y)*math.log1p(-q)))
        if not terms:
            raise ValueError('HEM requires known labels per training image')
        weights[i] = max(sum(terms)/len(terms),1e-8)
    return list(range(len(rows))),weights
