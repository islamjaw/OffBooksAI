import sys
sys.path.append('..')
import networkx as nx
from datetime import datetime
from agents.base_agent import BaseAgent

# Louvain community detection — install with: pip install python-louvain
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


class RingScout(BaseAgent):
    def __init__(self, graph_builder):
        super().__init__('Ring Scout')
        self.gb = graph_builder            # shared GraphBuilder instance
        self.flagged_rings = []            # history of all detected rings
        self.ring_counter = 0

        if LOUVAIN_AVAILABLE:
            self.log('Louvain community detection enabled (graph ML active).')
        else:
            self.log('python-louvain not installed — falling back to connected components. '
                     'Run: pip install python-louvain')

        # Detection rules — DefenseAI will append to this list
        self.rules = [
            'fan_out',
            'structuring',
            'circular',
            'velocity',
            'shared_metadata'
        ]

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    async def execute(self, input_data=None):
        """
        Scan the current graph for suspicious rings.
        Returns a list of ring dicts (may be empty if nothing found).
        """
        rings = self._scan()
        if rings:
            self.log(f'Detected {len(rings)} suspicious ring(s).')
        else:
            self.log('No suspicious rings detected.')
        return rings

    # ------------------------------------------------------------------
    # Main scan — runs all active rules
    # ------------------------------------------------------------------
    def _scan(self):
        graph = self.gb.graph
        if graph.number_of_nodes() == 0:
            return []

        undirected = graph.to_undirected()

        # ── Clustering: Louvain (graph ML) with fallback ──────────────
        if LOUVAIN_AVAILABLE and undirected.number_of_edges() > 0:
            # Louvain returns a dict: {node: community_id}
            # We invert it to get {community_id: set_of_nodes}
            partition = community_louvain.best_partition(undirected)
            community_map = {}
            for node, cid in partition.items():
                community_map.setdefault(cid, set()).add(node)
            components = list(community_map.values())
            self.log(f'Louvain found {len(components)} communities.')
        else:
            # Fallback: plain connected components
            components = list(nx.connected_components(undirected))
            self.log(f'Connected-components found {len(components)} clusters.')

        # ── Score each cluster with the active heuristic rules ────────
        rings = []
        for component in components:
            if len(component) < 2:
                continue

            score, patterns = self._score_component(component, graph)
            if score >= 50:                # threshold to flag
                self.ring_counter += 1
                ring = {
                    'ring_id': f'R_{self.ring_counter:03d}',
                    'accounts': list(component),
                    'suspicion_score': score,
                    'patterns': patterns,
                    'total_amount': self._component_volume(component, graph),
                    'timeframe_hours': self._component_timeframe(component, graph),
                    'timestamp': datetime.utcnow().isoformat(),
                    'cluster_method': 'louvain' if LOUVAIN_AVAILABLE else 'connected_components'
                }
                self.flagged_rings.append(ring)
                rings.append(ring)

        return rings

    # ------------------------------------------------------------------
    # Scoring logic — each rule contributes points
    # ------------------------------------------------------------------
    def _score_component(self, component, graph):
        score = 0
        patterns = []

        subgraph = graph.subgraph(component)

        if 'fan_out' in self.rules and self._check_fan_out(subgraph):
            score += 40
            patterns.append('fan_out')

        if 'structuring' in self.rules and self._check_structuring(subgraph):
            score += 30
            patterns.append('structuring')

        if 'circular' in self.rules and self._check_circular(subgraph):
            score += 40
            patterns.append('circular')

        if 'velocity' in self.rules and self._check_velocity(subgraph):
            score += 20
            patterns.append('velocity')

        if 'shared_metadata' in self.rules and self._check_shared_metadata(component):
            score += 25
            patterns.append('shared_metadata')

        # Cap at 100
        return min(score, 100), patterns

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------
    def _check_fan_out(self, subgraph):
        """
        Fan-out: one node sends to 3+ different recipients.
        Classic money-mule pattern.
        """
        for node in subgraph.nodes():
            out_degree = subgraph.out_degree(node)
            if out_degree >= 3:
                return True
        return False

    def _check_structuring(self, subgraph):
        """
        Structuring (smurfing): many transactions just under $500
        to avoid reporting thresholds.
        """
        suspicious_txns = 0
        for _, _, data in subgraph.edges(data=True):
            avg_amount = data.get('amount', 0) / max(data.get('count', 1), 1)
            if 400 <= avg_amount <= 499:
                suspicious_txns += 1
        return suspicious_txns >= 2

    def _check_circular(self, subgraph):
        """
        Circular routing: money goes A->B->C->...->A (wash trading).
        NetworkX can find simple cycles in a directed graph.
        """
        try:
            cycles = list(nx.simple_cycles(subgraph))
            return len(cycles) > 0
        except Exception:
            return False

    def _check_velocity(self, subgraph):
        """
        High velocity: cluster has 4+ total transactions AND 2+ distinct
        senders. Requires both so a single transfer does not auto-trip it.
        """
        total_txn_count = sum(
            data.get('count', 1)
            for _, _, data in subgraph.edges(data=True)
        )
        distinct_senders = sum(
            1 for n in subgraph.nodes()
            if subgraph.out_degree(n) > 0
        )
        return total_txn_count >= 4 and distinct_senders >= 2

    def _check_shared_metadata(self, component):
        """
        Shared device / IP across supposedly unrelated accounts.
        """
        metadata = self.gb.account_metadata
        devices = [
            metadata.get(acc, {}).get('device')
            for acc in component
            if metadata.get(acc, {}).get('device')
        ]
        ips = [
            metadata.get(acc, {}).get('ip')
            for acc in component
            if metadata.get(acc, {}).get('ip')
        ]
        # Flag if 2+ accounts share the same device or IP
        return (len(devices) != len(set(devices)) and len(devices) > 1) or \
               (len(ips) != len(set(ips)) and len(ips) > 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _component_volume(self, component, graph):
        """Total $ moved within a component."""
        subgraph = graph.subgraph(component)
        return sum(data.get('amount', 0) for _, _, data in subgraph.edges(data=True))

    def _component_timeframe(self, component, graph):
        """Rough timeframe in hours (placeholder — returns 1 if no timestamps)."""
        return 1.0

    def add_rule(self, rule_name):
        """Called by DefenseAI to extend detection coverage."""
        if rule_name not in self.rules:
            self.rules.append(rule_name)
            self.log(f'New detection rule added: {rule_name}')