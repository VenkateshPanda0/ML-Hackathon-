"""HMM + Viterbi route reconstruction (Newson & Krumm style) using classifier emission scores.

Hidden state  : one candidate road segment (arc id + projection fraction) for a GPS observation
Emission      : w * log P_classifier(candidate is the true arc | features), renormalised per observation
Transition    : -|network distance - straight-line distance| / beta   (network distance via Dijkstra)
                unreachable / implausible moves get a fixed "jump" penalty instead of -inf
Decoding      : Viterbi (max-sum); the chosen arcs are joined by shortest paths into a route.
"""
import numpy as np

DECODER_DEFAULTS = dict(
    beta=10.0,            # transition scale (m)
    emission_weight=1.0,  # weight of classifier log-probability
    route_factor=4.0,     # max network distance = route_factor * straight-line + route_slack
    route_slack=300.0,
    backtrack_tol=30.0,   # backward GPS jitter along the same arc treated as "stay" (m)
    jump_penalty=40.0,    # log-prob cost of moving between unconnected candidates
    stitch_factor=3.0,    # connecting path allowed up to stitch_factor * straight-line + stitch_slack
    stitch_slack=500.0,
    max_loop_len=1500.0,  # remove closed detours shorter than this (m)
)


def group_steps(cand, logp):
    steps = []
    for obs, g in cand.groupby("obs", sort=True):
        ix = g.index.to_numpy()
        steps.append(dict(obs=int(obs), i=int(g["pt"].iloc[0]), arcs=g["arc"].to_numpy(),
                          frac=g["frac"].to_numpy(), emis=logp[ix]))
    return steps


def normalised_log(p, eps=1e-6):
    return np.log(np.clip(p, eps, 1.0))


def transition(ctx, prev, cur, P):
    trip, graph = ctx.trip, ctx.graph
    arcs, L = trip.arcs, trip.arc_len
    gc = float(np.linalg.norm(trip.track_xy[cur["i"]] - trip.track_xy[prev["i"]]))
    limit = P["route_factor"] * gc + P["route_slack"]
    src = np.unique(arcs[prev["arcs"], 1])
    dist, _ = graph.sssp(src, limit)
    row = {n: r for r, n in enumerate(src)}
    route = np.full((len(prev["arcs"]), len(cur["arcs"])), np.inf)
    for a, (ea, fa) in enumerate(zip(prev["arcs"], prev["frac"])):
        route[a] = (1 - fa) * L[ea] + dist[row[arcs[ea, 1]], arcs[cur["arcs"], 0]] + cur["frac"] * L[cur["arcs"]]
        same = cur["arcs"] == ea
        back_m = (fa - cur["frac"]) * L[ea]
        route[a, same] = np.maximum(cur["frac"][same] - fa, 0) * L[ea] + \
            np.maximum(back_m[same] - P["backtrack_tol"], 0)
    logp = -np.abs(route - gc) / P["beta"]
    logp[route > limit] = -P["jump_penalty"]
    return logp


def viterbi(ctx, steps, P):
    """Return the chosen candidate index for every step."""
    w = P["emission_weight"]
    score = w * steps[0]["emis"]
    back = [None]
    for t in range(1, len(steps)):
        tot = score[:, None] + transition(ctx, steps[t - 1], steps[t], P)
        best = tot.argmax(0)
        back.append(best)
        score = tot[best, np.arange(tot.shape[1])] + w * steps[t]["emis"]
    states = [0] * len(steps)
    states[-1] = int(np.argmax(score))
    for t in range(len(steps) - 1, 0, -1):
        states[t - 1] = int(back[t][states[t]])
    return states


def stitch(ctx, steps, arc_seq, P):
    """Join the per-observation arcs into one route with bounded shortest paths."""
    trip, graph = ctx.trip, ctx.graph
    out = [int(arc_seq[0])]
    for t in range(1, len(arc_seq)):
        a, b = out[-1], int(arc_seq[t])
        if a == b:
            continue
        gc = np.linalg.norm(trip.track_xy[steps[t]["i"]] - trip.track_xy[steps[t - 1]["i"]])
        path = graph.shortest_arcs(int(trip.arcs[a, 1]), int(trip.arcs[b, 0]),
                                   limit=P["stitch_factor"] * gc + P["stitch_slack"])
        out.extend(path or [])          # None -> leave a gap (implausible connection)
        out.append(b)
    clean = [out[0]]
    for a in out[1:]:
        if a != clean[-1]:
            clean.append(a)
    return remove_loops(clean, trip.arcs, trip.arc_len, P["max_loop_len"])


def remove_loops(route, arcs, arc_len, max_loop_len):
    """Cut short closed detours (route returns to an already visited node), typically caused by GPS
    noise around junctions. Loops longer than max_loop_len are kept (may be genuine)."""
    out, seen, cum = [], {}, [0.0]
    for a in route:
        u = int(arcs[a, 0])
        if out and int(arcs[out[-1], 1]) != u:
            seen = {}
        if u in seen and cum[-1] - cum[seen[u]] <= max_loop_len:
            k = seen[u]
            for b in out[k:]:
                seen.pop(int(arcs[b, 0]), None)
            del out[k:]
            del cum[k + 1:]
        seen[u] = len(out)
        out.append(a)
        cum.append(cum[-1] + arc_len[a])
    return out


def decode(ctx, cand, proba, params=None):
    """Classifier probabilities -> (per-observation arcs, HMM route)."""
    P = {**DECODER_DEFAULTS, **(params or {})}
    steps = group_steps(cand, normalised_log(proba))
    states = viterbi(ctx, steps, P)
    obs_arcs = np.array([s["arcs"][k] for s, k in zip(steps, states)])
    return obs_arcs, stitch(ctx, steps, obs_arcs, P), steps


def classifier_only(ctx, cand, proba, params=None):
    """Top-1 candidate per observation, no sequence model (baseline for the HMM comparison)."""
    P = {**DECODER_DEFAULTS, **(params or {})}
    steps = group_steps(cand, normalised_log(proba))
    obs_arcs = np.array([s["arcs"][int(np.argmax(s["emis"]))] for s in steps])
    return obs_arcs, stitch(ctx, steps, obs_arcs, P), steps
