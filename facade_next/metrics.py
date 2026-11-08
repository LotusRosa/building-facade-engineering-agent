"""Known-label metrics/calibration, without numpy or model execution."""
import math
from .contracts import CLASSES


def _check(rows, probs):
    if len(rows) != len(probs) or not rows:
        raise ValueError('Prediction row count mismatch/empty')
    for p in probs:
        if len(p) != 3 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in p):
            raise ValueError('Expected finite probabilities [N,3] in [0,1]')


def average_precision(pairs):
    positives = sum(y for y,p in pairs)
    if not positives:
        return None
    ordered = sorted(pairs,key=lambda x:-x[1])
    tp, seen, ap, i = 0, 0, 0., 0
    while i < len(ordered):
        j = i+1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        new_tp = sum(x[0] for x in ordered[i:j])
        tp += new_tp
        seen += j-i
        ap += new_tp/positives * tp/seen
        i = j
    return ap


def _counts(pairs, threshold):
    tp = sum(y == 1 and p >= threshold for y,p in pairs)
    fp = sum(y == 0 and p >= threshold for y,p in pairs)
    fn = sum(y == 1 and p < threshold for y,p in pairs)
    return tp,fp,fn


def evaluate(rows, probs, thresholds):
    _check(rows,probs)
    results = {}
    for j,c in enumerate(CLASSES):
        pairs = [(r['labels'][j],p[j]) for r,p in zip(rows,probs) if r['labels'][j] is not None]
        tp,fp,fn = _counts(pairs, thresholds[j])
        results[c] = dict(AP=average_precision(pairs), precision=tp/(tp+fp) if tp+fp else None,
                          recall=tp/(tp+fn) if tp+fn else None,
                          F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                          TP=tp,FP=fp,FN=fn,known=len(pairs),positive=sum(y for y,p in pairs), threshold=thresholds[j])
    aps = [x['AP'] for x in results.values() if x['AP'] is not None]
    return {'classes':results, 'macro_AP':sum(aps)/len(aps) if aps else None, 'AP_defined_classes':len(aps), 'images':len(rows)}


def calibrate(rows, probs, parent=None):
    _check(rows,probs)
    result, status = [], []
    for j in range(3):
        pairs = [(r['labels'][j],p[j]) for r,p in zip(rows,probs) if r['labels'][j] is not None]
        if {y for y,p in pairs} != {0,1}:
            if parent is None:
                raise ValueError('Initial Val requires known positive and negative examples for every class')
            result.append(parent[j])
            status.append('inherited_no_positive_or_negative')
            continue
        best, chosen = -1., None
        for n in range(181):
            t = round(.05 + .005*n, 6)
            tp,fp,fn = _counts(pairs,t)
            f1 = 2*tp/max(2*tp+fp+fn,1)
            if f1 > best:
                best,chosen = f1,t
        result.append(chosen)
        status.append('calibrated_on_val')
    return result,status


def scene_metrics(rows, probs, thresholds):
    result = {'merged':evaluate(rows,probs,thresholds), 'scenes':{}}
    for scene in sorted({r['scene'] for r in rows}):
        indices = [i for i,r in enumerate(rows) if r['scene'] == scene]
        result['scenes'][scene] = evaluate([rows[i] for i in indices], [probs[i] for i in indices], thresholds)
    return result
