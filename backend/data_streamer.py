"""
DataStreamer — streams real transactions from the Kaggle fraud dataset.

Dataset columns:
  transaction_id, amount, transaction_hour, merchant_category,
  foreign_transaction, location_mismatch, device_trust_score,
  velocity_last_24h, cardholder_age, is_fraud

Provides:
  get_fraud_ring(size)   — real fraud transactions structured as a coordinated ring
  get_hard_cases(n)      — stealthiest real fraud cases (low amount, high device trust)
  get_legit_batch(n)     — real legitimate transactions for graph background noise
  get_stats()            — dataset statistics for the /stats endpoint
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataStreamer:
    def __init__(self, csv_path: str = 'data/creditcard.csv'):
        print(f'[DataStreamer] Loading: {csv_path}')
        self.df    = pd.read_csv(csv_path)
        self.fraud = self.df[self.df['is_fraud'] == 1].reset_index(drop=True)
        self.legit = self.df[self.df['is_fraud'] == 0].reset_index(drop=True)

        print(
            f'[DataStreamer] {len(self.df):,} transactions — '
            f'{len(self.fraud)} fraud ({len(self.fraud)/len(self.df)*100:.2f}%), '
            f'{len(self.legit):,} legitimate'
        )

        self._ring_counter    = 0
        self._account_cache   = {}   # stable account IDs across runs

    # ── Public API ──────────────────────────────────────────────────────
    def get_fraud_ring(self, size: int = 6) -> dict:
        """
        Build a real fraud ring from the dataset.
        Structure: fan-out from one source + one return hop to create
        a partial cycle — exhibits multiple ring patterns simultaneously.
        """
        self._ring_counter += 1
        sample = self.fraud.sample(min(size, len(self.fraud)))
        rows   = [row for _, row in sample.iterrows()]

        src_id = f'FRAUD_SRC_{self._ring_counter:03d}'
        txns   = []

        # Fan-out: source → mule accounts
        for i, row in enumerate(rows):
            dst_id = f'MULE_{self._ring_counter:03d}_{i:02d}'
            txns.append(self._row_to_txn(row, src_id, dst_id))

        # Add partial cycle: mule_0 → mule_1 (creates circular signature)
        if len(rows) >= 2:
            txns.append({
                **self._row_to_txn(rows[1], f'MULE_{self._ring_counter:03d}_00',
                                             f'MULE_{self._ring_counter:03d}_01'),
                'amount': float(rows[1]['amount']) * 0.88
            })

        return {
            'strategy':     f'real_fraud_ring_{self._ring_counter}',
            'rationale':    'Real fraudulent transactions from Kaggle dataset',
            'transactions': txns,
            'source':       'Kaggle Credit Card Fraud Dataset',
            'true_fraud':   True,
        }

    def get_hard_cases(self, n: int = 5) -> dict:
        """
        Hardest-to-detect fraud: low amount, high device trust, no location flag.
        These are the cases that slip past simple rule-based systems.
        Used for difficulty 4-5 attacks.
        """
        self._ring_counter += 1
        hard = self.fraud[
            (self.fraud['amount']             < 150) &
            (self.fraud['device_trust_score'] >= 50)  &
            (self.fraud['location_mismatch']  == 0)
        ]
        if len(hard) == 0:
            hard = self.fraud

        sample = hard.sample(min(n, len(hard)))
        txns   = []
        for i, (_, row) in enumerate(sample.iterrows()):
            src = f'STEALTH_SRC_{self._ring_counter:03d}'
            dst = f'STEALTH_DST_{self._ring_counter:03d}_{i:02d}'
            txns.append(self._row_to_txn(row, src, dst))

        return {
            'strategy':     f'stealth_real_fraud_{self._ring_counter}',
            'rationale':    'Low-amount fraud with high device trust — evades simple heuristics',
            'transactions': txns,
            'source':       'Kaggle Credit Card Fraud Dataset (hard cases)',
            'true_fraud':   True,
        }

    def get_legit_batch(self, n: int = 15) -> list:
        """
        Real legitimate transactions for background graph noise.
        Uses isolated pairs so they don't accidentally trip Ring Scout.
        """
        sample   = self.legit.sample(min(n, len(self.legit)))
        accounts = [f'CUST_{i:03d}' for i in range(1, 35)]
        txns     = []
        used_pairs = set()

        for _, row in sample.iterrows():
            # Pick an isolated pair not used in this batch
            for _ in range(20):
                src = random.choice(accounts)
                dst = random.choice(accounts)
                if src != dst and (src, dst) not in used_pairs:
                    used_pairs.add((src, dst))
                    break
            txns.append(self._row_to_txn(row, src, dst))

        return txns

    def get_stats(self) -> dict:
        return {
            'total':             len(self.df),
            'fraud_count':       len(self.fraud),
            'legit_count':       len(self.legit),
            'fraud_rate_pct':    round(len(self.fraud) / len(self.df) * 100, 3),
            'features':          list(self.df.columns),
            'avg_fraud_amount':  round(float(self.fraud['amount'].mean()), 2),
            'avg_legit_amount':  round(float(self.legit['amount'].mean()), 2),
            'max_fraud_amount':  round(float(self.fraud['amount'].max()), 2),
            'merchant_categories': self.df['merchant_category'].unique().tolist()
                                   if 'merchant_category' in self.df.columns else [],
        }

    # ── Row → transaction dict ──────────────────────────────────────────
    def _row_to_txn(self, row, src_id: str, dst_id: str) -> dict:
        hour = int(row.get('transaction_hour', 12))
        base = datetime.utcnow().replace(microsecond=0)
        ts   = (base - timedelta(hours=random.randint(0, 4),
                                  minutes=random.randint(0, 59))).isoformat()
        return {
            'from':                 str(src_id),
            'to':                   str(dst_id),
            'amount':               float(row['amount']),
            'transaction_hour':     hour,
            'merchant_category':    str(row.get('merchant_category', 'Unknown')),
            'foreign_transaction':  int(row.get('foreign_transaction', 0)),
            'location_mismatch':    int(row.get('location_mismatch',   0)),
            'device_trust_score':   float(row.get('device_trust_score', 50)),
            'velocity_last_24h':    int(row.get('velocity_last_24h',    1)),
            'cardholder_age':       int(row.get('cardholder_age',       35)),
            'true_label':           int(row.get('is_fraud',             0)),
            'timestamp':            ts,
            'device':               f'device_{int(row.get("device_trust_score", 50)):03d}',
            'ip':                   self._make_ip(row),
            'transaction_id':       int(row.get('transaction_id',       0)),
            'delay_minutes':        0,
        }

    def _make_ip(self, row) -> str:
        if int(row.get('foreign_transaction', 0)):
            return f'185.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}'
        return f'192.168.{random.randint(1,10)}.{random.randint(1,254)}'