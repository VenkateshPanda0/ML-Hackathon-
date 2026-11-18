"""Directed road graph over nodes, with arc lookup for path reconstruction."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class RoadGraph:
    def __init__(self, trip):
        arcs, L = trip.arcs, trip.arc_len
        n = len(trip.nodes_xy)
        # keep the shortest arc for each (u, v) pair (parallel arcs are rare)
        order = np.argsort(-L)
        self.pair_arc = {}
        for a in order:
            self.pair_arc[(int(arcs[a, 0]), int(arcs[a, 1]))] = int(a)
        u = np.array([k[0] for k in self.pair_arc])
        v = np.array([k[1] for k in self.pair_arc])
        w = np.array([L[a] for a in self.pair_arc.values()])
        self.G = csr_matrix((w, (u, v)), shape=(n, n))
        self.n = n

    def sssp(self, sources, limit):
        """Shortest distances + predecessors from each source node, explored up to `limit` metres."""
        dist, pred = dijkstra(self.G, directed=True, indices=sources, limit=limit,
                              return_predecessors=True)
        return dist, pred

    def path_arcs(self, pred_row, src, dst):
        """Arc ids along the shortest path src -> dst given a predecessor row (None if unreachable)."""
        if src == dst:
            return []
        nodes = [dst]
        cur = dst
        while cur != src:
            cur = pred_row[cur]
            if cur < 0:
                return None
            nodes.append(cur)
        nodes.reverse()
        return [self.pair_arc[(int(nodes[i]), int(nodes[i + 1]))] for i in range(len(nodes) - 1)]

    def shortest_arcs(self, src, dst, limit=np.inf):
        if src == dst:
            return []
        _, pred = dijkstra(self.G, directed=True, indices=[src], limit=limit, return_predecessors=True)
        return self.path_arcs(pred[0], src, dst)
