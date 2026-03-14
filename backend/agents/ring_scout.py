"""
RingScout — graph-based fraud ring detection engine.

Two-layer detection:
  Layer 1 — ML consensus: checks if enough accounts in the cluster
             were independently flagged by TransactionScorer.
  Layer 2 — Graph topology: 7 heuristic rules (fan_out, structuring,
             circular, velocity, shared_metadata, layering, pagerank_anomaly)
             + any rules added at runtime by DefenseAI.

A cluster must score >= 50 to be flagged as a ring.
"""
import sys
sys.path.append('..')
import networkx as nx
from datetime import datetime
from agents.base_agent import BaseAgent

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


RULE_WEIGHTS = {
    'ml_consensus':     50,   # ML + graph agreement — highest confidence
    'circular':         45,   # wash trading
    'layering':         40,   # 3+ hop chain
    'fan_out':          35,   # 1 source → many recipients
    'pagerank_anomaly': 35,   # anomalously central node
    'structuring':      30,   # amounts just below $500
    'device_anomaly':   28,   # low device trust + high velocity
    'shared_metadata':  25,   # shared device / IP
    'location_cluster': 22,   # multiple foreign/mismatched accounts in cluster
    'velocity':         15,   # high transaction count
}


class RingScout(BaseAgent):
    def __init__(self, graph_builder):
        super().__init__('Ring Scout')
        self.gb            = graph_builder
        self.flagged_rings = []
        self.ring_counter  = 0
        self.rules         = list(RULE_WEIGHTS.keys())

        mode = 'Louvain' if LOUVAIN_AVAILABLE else 'connected-components fallback'
        self.log(f'Initialized — community detection: {mode}')

    # ── BaseAgent contract ──────────────────────────────────────────────
    async def execute(self, input_data=None) -> list:
        self.gb.enrich_graph_features()
        rings = self._scan()
        self.log(f'Detected {len(rings)} ring(s).' if rings else 'No rings detected.')
        return rings

    # ── Main scan ───────────────────────────────────────────────────────
    def _scan(self) -> list:
        graph = self.gb.graph
        if graph.number_of_nodes() == 0:
            return []

        undirected = graph.to_undirected()

        if LOUVAIN_AVAILABLE and undirected.number_of_edges() > 0:
            partition     = community_louvain.best_partition(undirected)
            community_map = {}
            for node, cid in partition.items():
                community_map.setdefault(cid, set()).add(node)
            components = list(community_map.values())
            self.log(f'Louvain: {len(components)} communities')
        else:
            components = list(nx.connected_components(undirected))
            self.log(f'Connected-components: {len(components)} clusters')

        rings = []
        for component in components:
            if len(component) < 2:
                continue
            score, patterns = self._score_component(component, graph)
            if score >= 50:
                self.ring_counter += 1
                high_pr = self.gb.get_high_pagerank_nodes()
                ring = {
                    'ring_id':          f'R_{self.ring_counter:03d}',
                    'accounts':         list(component),
                    'suspicion_score':  score,
                    'patterns':         patterns,
                    'total_amount':     self._component_volume(component, graph),
                    'timeframe_hours':  self._component_timeframe(component, graph),
                    'timestamp':        datetime.utcnow().isoformat(),
                    'cluster_method':   'louvain' if LOUVAIN_AVAILABLE
                                        else 'connected_components',
                    'high_pr_nodes':    high_pr,
                    'node_centrality':  self.gb.get_node_centrality_dict(),
                    'ml_active':        self._ml_active(component),
                    'heuristic_score':  score,
                }
                # Attach ML probability if available
                ml_prob = self._cluster_ml_probability(component, graph)
                if ml_prob is not None:
                    ring['ml_probability'] = ml_prob
                    # Blend ML probability into suspicion score
                    ring['suspicion_score'] = min(
                        int(score * 0.6 + ml_prob * 100 * 0.4), 100
                    )

                self.flagged_rings.append(ring)
                rings.append(ring)
        return rings

    # ── Scoring ─────────────────────────────────────────────────────────
    def _score_component(self, component: set, graph: nx.DiGraph):
        score    = 0
        patterns = []
        subgraph = graph.subgraph(component)

        checks = {
            'ml_consensus':     lambda: self._check_ml_consensus(component, graph),
            'fan_out':          lambda: self._check_fan_out(subgraph),
            'structuring':      lambda: self._check_structuring(subgraph),
            'circular':         lambda: self._check_circular(subgraph),
            'velocity':         lambda: self._check_velocity(subgraph),
            'shared_metadata':  lambda: self._check_shared_metadata(component),
            'layering':         lambda: self._check_layering(subgraph),
            'pagerank_anomaly': lambda: self._check_pagerank_anomaly(subgraph),
            'device_anomaly':   lambda: self._check_device_anomaly(component, graph),
            'location_cluster': lambda: self._check_location_cluster(component, graph),
        }

        for rule, fn in checks.items():
            if rule not in self.rules:
                continue
            try:
                if fn():
                    w = RULE_WEIGHTS.get(rule, 20)
                    score   += w
                    patterns.append(rule)
            except Exception as e:
                self.log(f'Rule {rule} error: {e}')

        return min(score, 100), patterns

    # ── Rule implementations ────────────────────────────────────────────
    def _check_ml_consensus(self, component: set, graph: nx.DiGraph) -> bool:
        """3+ accounts in the cluster were ML-flagged (fraud_score > 60)."""
        count = sum(
            1 for node in component
            if graph.nodes[node].get('ml_flagged', False)
        )
        return count >= 3

    def _check_fan_out(self, sg: nx.DiGraph) -> bool:
        """One node sends to 3+ different recipients."""
        return any(sg.out_degree(n) >= 3 for n in sg.nodes())

    def _check_structuring(self, sg: nx.DiGraph) -> bool:
        """2+ transactions with average amount $400-$499 (below reporting threshold)."""
        suspicious = sum(
            1 for _, _, d in sg.edges(data=True)
            if 400 <= (d.get('amount', 0) / max(d.get('count', 1), 1)) <= 499
        )
        return suspicious >= 2

    def _check_circular(self, sg: nx.DiGraph) -> bool:
        """Any directed cycle (wash trading / A→B→…→A)."""
        try:
            return len(list(nx.simple_cycles(sg))) > 0
        except Exception:
            return False

    def _check_velocity(self, sg: nx.DiGraph) -> bool:
        """4+ total transactions with 2+ distinct senders."""
        total    = sum(d.get('count', 1) for _, _, d in sg.edges(data=True))
        senders  = sum(1 for n in sg.nodes() if sg.out_degree(n) > 0)
        return total >= 4 and senders >= 2

    def _check_shared_metadata(self, component: set) -> bool:
        """2+ accounts share the same device fingerprint or IP."""
        meta    = self.gb.account_metadata
        devices = [meta.get(a, {}).get('device') for a in component
                   if meta.get(a, {}).get('device')]
        ips     = [meta.get(a, {}).get('ip') for a in component
                   if meta.get(a, {}).get('ip')]
        dev_dup = len(devices) > 1 and len(set(devices)) < len(devices)
        ip_dup  = len(ips)     > 1 and len(set(ips))     < len(ips)
        return dev_dup or ip_dup

    def _check_layering(self, sg: nx.DiGraph) -> bool:
        """2+ paths of length >= 3 (classic placement-layering-integration chain)."""
        long_paths = 0
        nodes = list(sg.nodes())
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                try:
                    if nx.shortest_path_length(sg, src, dst) >= 3:
                        long_paths += 1
                        if long_paths >= 2:
                            return True
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return False

    def _check_pagerank_anomaly(self, sg: nx.DiGraph) -> bool:
        """One node is 3× more central than the cluster mean (collection account)."""
        if sg.number_of_nodes() < 3:
            return False
        try:
            pr     = nx.pagerank(sg, alpha=0.85, max_iter=200)
            vals   = list(pr.values())
            mean   = sum(vals) / len(vals)
            return mean > 0 and any(v > mean * 3.0 for v in vals)
        except Exception:
            return False

    def _check_device_anomaly(self, component: set, graph: nx.DiGraph) -> bool:
        """
        Multiple accounts with low device trust score (<40) AND high velocity (>5).
        Consistent with scripted / automated fraud tools.
        """
        count = sum(
            1 for node in component
            if (graph.nodes[node].get('avg_device_trust', 100) < 40 and
                graph.nodes[node].get('max_velocity',       0) > 5)
        )
        return count >= 2

    def _check_location_cluster(self, component: set, graph: nx.DiGraph) -> bool:
        """
        3+ accounts in the cluster have location mismatches or foreign transactions.
        Organized fraud rings often operate from a single geographic location
        targeting cards registered elsewhere.
        """
        count = sum(
            1 for node in component
            if (graph.nodes[node].get('location_mismatch_count', 0) > 0 or
                graph.nodes[node].get('foreign_count',            0) > 0)
        )
        return count >= 3

    # ── Helpers ─────────────────────────────────────────────────────────
    def _cluster_ml_probability(self, component: set, graph: nx.DiGraph):
        """Average ML fraud probability across accounts that have one."""
        scores = [
            graph.nodes[n].get('fraud_score', 0) / 100.0
            for n in component
            if graph.nodes[n].get('ml_flagged', False)
        ]
        return round(sum(scores) / len(scores), 3) if scores else None

    def _ml_active(self, component: set) -> bool:
        return any(
            self.gb.graph.nodes[n].get('fraud_score', 0) > 0
            for n in component
        )

    def _component_volume(self, component: set, graph: nx.DiGraph) -> float:
        sg = graph.subgraph(component)
        return round(sum(d.get('amount', 0) for _, _, d in sg.edges(data=True)), 2)

    def _component_timeframe(self, component: set, graph: nx.DiGraph) -> float:
        return 1.0   # placeholder — real timestamps need parsing

    def add_rule(self, rule_name: str, weight: int = 25):
        if rule_name not in self.rules:
            self.rules.append(rule_name)
            RULE_WEIGHTS[rule_name] = weight
            self.log(f'New rule: {rule_name} (weight={weight})')