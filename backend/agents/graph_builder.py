import sys
sys.path.append('..')
import networkx as nx
from datetime import datetime
from agents.base_agent import BaseAgent


class GraphBuilder(BaseAgent):
    def __init__(self):
        super().__init__('Graph Builder')
        self.graph = nx.DiGraph()          # directed: A -> B means A paid B
        self.transactions = []             # full history of every transaction
        self.account_metadata = {}         # ip, device, location per account

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    async def execute(self, input_data):
        """
        input_data can be:
          - a single transaction dict
          - {'transactions': [...]} batch
          - an attack plan dict from FraudGPT (has 'transactions' key)
        Returns updated graph stats.
        """
        if 'transactions' in input_data:
            txns = input_data['transactions']
            for t in txns:
                self.add_transaction(t)
            self.log(f'Ingested {len(txns)} transactions. '
                     f'Graph: {self.graph.number_of_nodes()} nodes, '
                     f'{self.graph.number_of_edges()} edges')
        else:
            self.add_transaction(input_data)

        return self.get_stats()

    # ------------------------------------------------------------------
    # Core graph operations
    # ------------------------------------------------------------------
    def add_transaction(self, txn):
        """
        txn = {
            'from': 'ACC_A', 'to': 'ACC_B',
            'amount': 500, 'delay_minutes': 0,
            'timestamp': '...',   # optional
            'ip': '...',          # optional metadata
            'device': '...'       # optional metadata
        }
        """
        src = txn.get('from', 'UNKNOWN')
        dst = txn.get('to', 'UNKNOWN')
        amount = txn.get('amount', 0)
        ts = txn.get('timestamp', datetime.utcnow().isoformat())

        # Add nodes if new
        for acc in [src, dst]:
            if acc not in self.graph:
                self.graph.add_node(acc, total_sent=0, total_received=0,
                                    tx_count=0, first_seen=ts, last_seen=ts)

        # Update node stats
        self.graph.nodes[src]['total_sent'] += amount
        self.graph.nodes[src]['tx_count'] += 1
        self.graph.nodes[src]['last_seen'] = ts

        self.graph.nodes[dst]['total_received'] += amount
        self.graph.nodes[dst]['tx_count'] += 1
        self.graph.nodes[dst]['last_seen'] = ts

        # Add / update edge (accumulate if multiple txns between same pair)
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]['amount'] += amount
            self.graph[src][dst]['count'] += 1
            self.graph[src][dst]['timestamps'].append(ts)
        else:
            self.graph.add_edge(src, dst, amount=amount, count=1,
                                timestamps=[ts])

        # Store optional metadata
        for field in ('ip', 'device', 'location'):
            val = txn.get(field)
            if val:
                self.account_metadata.setdefault(src, {})[field] = val

        self.transactions.append({**txn, 'timestamp': ts})

    def get_subgraph(self, accounts):
        """Return the subgraph induced by a list of account IDs."""
        nodes = [a for a in accounts if a in self.graph]
        return self.graph.subgraph(nodes).copy()

    def get_neighbors(self, account):
        """All accounts directly connected to this one (in or out)."""
        if account not in self.graph:
            return []
        return list(set(
            list(self.graph.predecessors(account)) +
            list(self.graph.successors(account))
        ))

    def get_stats(self):
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'transactions': len(self.transactions),
            'accounts': list(self.graph.nodes())
        }

    def reset(self):
        """Clear everything — used between battle rounds."""
        self.graph.clear()
        self.transactions.clear()
        self.account_metadata.clear()
        self.log('Graph reset.')

    # ------------------------------------------------------------------
    # Serialise for frontend (Cytoscape.js format)
    # ------------------------------------------------------------------
    def to_cytoscape(self, highlight_accounts=None):
        """
        Returns a dict with 'nodes' and 'edges' lists that Cytoscape.js
        can consume directly.
        highlight_accounts: set of account IDs to mark as suspicious (red)
        """
        highlight_accounts = set(highlight_accounts or [])
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                'data': {
                    'id': node_id,
                    'label': node_id,
                    'total_sent': data.get('total_sent', 0),
                    'total_received': data.get('total_received', 0),
                    'tx_count': data.get('tx_count', 0),
                    'suspicious': node_id in highlight_accounts
                }
            })

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            edges.append({
                'data': {
                    'id': f'{src}->{dst}',
                    'source': src,
                    'target': dst,
                    'amount': data.get('amount', 0),
                    'count': data.get('count', 1)
                }
            })

        return {'nodes': nodes, 'edges': edges}