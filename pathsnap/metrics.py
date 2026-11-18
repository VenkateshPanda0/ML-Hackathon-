"""Metrics: candidate classification, observation-level, route continuity and route (sequence) level."""
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0),
        macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
        log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    )


def top1_accuracy(cand, p):
    """Per GPS observation: is the highest-scoring candidate the reference arc?"""
    df = cand[["trip", "obs", "label"]].copy()
    df["p"] = p
    top = df.loc[df.groupby(["trip", "obs"])["p"].idxmax()]
    return float(top["label"].mean())


def candidate_miss_rate(cand):
    """Share of observations whose reference arc is not in the candidate set."""
    return float(1 - cand.groupby(["trip", "obs"])["label"].max().mean())


def pointwise_accuracy(obs_arcs, ref_by_obs):
    return float(np.mean(np.asarray(obs_arcs) == np.asarray(ref_by_obs)))


def continuity(arc_seq, arcs):
    """Share of consecutive changes of predicted arc that are physically adjacent (to_node == from_node)."""
    seq = [a for i, a in enumerate(arc_seq) if i == 0 or a != arc_seq[i - 1]]
    if len(seq) < 2:
        return 1.0
    ok = [arcs[seq[i], 1] == arcs[seq[i + 1], 0] for i in range(len(seq) - 1)]
    return float(np.mean(ok))


def route_continuity(route, arcs):
    """Share of consecutive arcs in the FINAL returned route that are physically adjacent
    (to_node of one arc == from_node of the next). Unlike `continuity`, which looks at the raw
    per-observation picks, this measures the stitched route the API actually returns."""
    route = list(map(int, route))
    if len(route) < 2:
        return 1.0
    return float(np.mean([arcs[route[i], 1] == arcs[route[i + 1], 0] for i in range(len(route) - 1)]))


def route_metrics(pred, true, arc_len):
    cp, ct = Counter(map(int, pred)), Counter(map(int, true))
    L = lambda c: sum(arc_len[a] * n for a, n in c.items())
    li, lp, lt = L(cp & ct), L(cp), L(ct)
    prec = li / lp if lp else 0.0
    rec = li / lt if lt else 0.0
    return dict(
        route_precision=prec, route_recall=rec,
        route_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        route_mismatch=(L(cp - ct) + L(ct - cp)) / lt if lt else np.nan,
        exact_match=float(list(map(int, pred)) == list(map(int, true))),
    )
