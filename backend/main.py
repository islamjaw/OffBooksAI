"""
SyndicateAI — main FastAPI application.

Three-layer fraud detection:
  1. TransactionScorer (ML)  — per-transaction XGBoost fraud probability
  2. GraphBuilder            — live transaction network with centrality metrics
  3. RingScout               — graph topology + ML consensus ring detection

Red team:
  FraudGPT      — generates adversarial attacks (Attack Mode tab uses DataStreamer
                   for real data; Battle Mode tab always uses FraudGPT with LLM)
  DefenseAI     — proposes new rules when attacks evade detection

Battle Mode agents are separate instances with their own memory so
the two tabs never interfere with each other.
"""
import sys
import asyncio
import json
import random
import traceback
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append('.')

# ── numpy sanitizer — must be defined BEFORE any imports that use it ─────
import numpy as np

def sanitize(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ── Debug helper ──────────────────────────────────────────────────────────
def _debug(label: str, data: dict = None):
    """Print a clear labelled block to the terminal for debugging."""
    sep = '─' * 55
    print(f'\n{sep}')
    print(f'[DEBUG] {label}')
    if data:
        for k, v in data.items():
            val = str(v)
            if len(val) > 140:
                val = val[:140] + '...'
            print(f'  {k:<28} {val}')
    print(sep)

# ── Agent imports ─────────────────────────────────────────────────────────
from agents.graph_builder       import GraphBuilder
from agents.ring_scout          import RingScout
from agents.investigation_agent import InvestigationAgent, GOVERNANCE_LOG
from agents.fraud_gpt           import FraudGPT
from agents.defense_ai          import DefenseAI
from agents.transaction_scorer  import TransactionScorer
from data_streamer              import DataStreamer

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title='SyndicateAI', version='2.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*']
)

# ── Shared infrastructure ─────────────────────────────────────────────────
graph_builder = GraphBuilder()
ring_scout    = RingScout(graph_builder)
investigation = InvestigationAgent()
scorer        = TransactionScorer()

# ── Attack Mode agents (Tab 1) ────────────────────────────────────────────
# Uses DataStreamer for real Kaggle fraud data when available.
# FraudGPT here is only used when real data mode is OFF.
fraud_gpt  = FraudGPT()
defense_ai = DefenseAI(ring_scout)

# ── Battle Mode agents (Tab 2) ────────────────────────────────────────────
# Always uses FraudGPT with LLM — never DataStreamer for the attack.
# DataStreamer only seeds legitimate background traffic.
# Completely separate memory from Attack Mode so the two tabs never interfere.
battle_fraud_gpt  = FraudGPT()
battle_defense_ai = DefenseAI(ring_scout)  # shares ring_scout so new rules apply globally

# ── Real data ─────────────────────────────────────────────────────────────
try:
    data_streamer  = DataStreamer('data/creditcard.csv')
    REAL_DATA_MODE = True
    print('[Main] Real data mode ENABLED')
except Exception as e:
    data_streamer  = None
    REAL_DATA_MODE = False
    print(f'[Main] Real data mode DISABLED ({e})')

# ── Attack Mode state (Tab 1) ─────────────────────────────────────────
attack_state = {
    'round':            0,
    'attacks_launched': 0,
    'detections':       0,
    'evasions':         0,
    'rules_added':      0,
    'true_positives':   0,
    'false_positives':  0,
    'false_negatives':  0,
    'log':              [],
    'last_attack':      None,
    'last_detected':    False,
    'last_detection_reason': '',
}

# ── Battle Mode state (Tab 2) ─────────────────────────────────────────
battle_state = {
    'running':          False,
    'round':            0,
    'attacks_launched': 0,
    'detections':       0,
    'evasions':         0,
    'rules_added':      0,
    'true_positives':   0,
    'false_positives':  0,
    'false_negatives':  0,
    'log':              [],
    'last_attack':      None,
    'last_detected':    False,
    'last_detection_reason': '',
}

# ── Battle Mode per-round state ───────────────────────────────────────────
battle_round_state = {
    'last_attack':           None,
    'last_detected':         False,
    'last_detection_reason': '',
    'difficulty':            1,
}

_LEGIT_ACCOUNTS = [f'CUST_{i:03d}' for i in range(1, 21)]
noise_state     = {'running': False, 'tx_count': 0}


# ── Background noise ──────────────────────────────────────────────────────
async def _noise_loop():
    while noise_state['running']:
        for _ in range(random.randint(1, 3)):
            src, dst = random.sample(_LEGIT_ACCOUNTS, 2)
            graph_builder.add_transaction({
                'from': src, 'to': dst,
                'amount':              round(random.uniform(15, 50), 2),
                'delay_minutes':       0,
                'ip':                  f'192.168.{random.randint(1,10)}.{random.randint(1,255)}',
                'device':              f'device_{random.randint(1, 15):02d}',
                'device_trust_score':  random.randint(70, 100),
                'location_mismatch':   0,
                'foreign_transaction': 0,
                'velocity_last_24h':   random.randint(1, 3),
            })
            noise_state['tx_count'] += 1
        await asyncio.sleep(2)


# ── Helpers ───────────────────────────────────────────────────────────────
def _log(message: str, kind: str = 'info', state: dict = None):
    if state is None:
        state = battle_state   # default to battle log for auto-battle loop
    entry = {
        'time':    datetime.utcnow().strftime('%H:%M:%S'),
        'message': message,
        'kind':    kind,
    }
    state['log'].append(entry)
    print(f"[{'BATTLE' if state is battle_state else 'ATTACK'}] {entry['time']} [{kind.upper()}] {message}")
    return entry


def _update_accuracy(attack: dict, detected: bool, state: dict = None):
    if state is None:
        state = battle_state
    if not REAL_DATA_MODE:
        return
    is_fraud = bool(attack.get('true_fraud', False))
    if   is_fraud and     detected: state['true_positives']  += 1
    elif is_fraud and not detected: state['false_negatives'] += 1
    elif not is_fraud and detected: state['false_positives'] += 1


def _score_transactions(transactions: list) -> list:
    """Score transactions with ML model. Adds fraud_score + ml_flagged to each dict."""
    try:
        if scorer.trained:
            scored = scorer.score_batch(transactions)
        else:
            scored = [scorer.score_transaction(t) for t in transactions]
        return [sanitize(t) for t in scored]
    except Exception as e:
        _debug('SCORING ERROR', {'error': str(e)})
        return [{**t, 'fraud_score': 0, 'ml_flagged': False, 'ml_source': 'error'} for t in transactions]


# ════════════════════════════════════════════════════════════════════════════
# ATTACK MODE (Tab 1) — run_one_round
# Uses real Kaggle fraud data (DataStreamer) when available,
# falls back to FraudGPT LLM when not.
# ════════════════════════════════════════════════════════════════════════════
async def run_one_round(difficulty: int = 1) -> dict:
    graph_builder.reset()

    # ── Seed legitimate background traffic ────────────────────────────
    if REAL_DATA_MODE and data_streamer:
        legit_txns   = data_streamer.get_legit_batch(n=15)
        scored_legit = _score_transactions(legit_txns)
        for txn in scored_legit:
            graph_builder.add_transaction(txn)
    else:
        _LEGIT = [f'CUST_{i:03d}' for i in range(1, 21)]
        pairs_used, seeded, attempts = set(), 0, 0
        while seeded < 8 and attempts < 40:
            attempts += 1
            src, dst = random.sample(_LEGIT, 2)
            pair     = tuple(sorted([src, dst]))
            if pair in pairs_used:
                continue
            pairs_used.add(pair)
            graph_builder.add_transaction({
                'from': src, 'to': dst,
                'amount':             round(random.uniform(15, 50), 2),
                'delay_minutes':      0,
                'device_trust_score': random.randint(70, 100),
            })
            seeded += 1

    # ── 1. Get attack ─────────────────────────────────────────────────
    if REAL_DATA_MODE and data_streamer:
        attack = (data_streamer.get_hard_cases(n=5)
                  if difficulty >= 4 else
                  data_streamer.get_fraud_ring(size=6))
    else:
        was_detected = (attack_state['last_attack'] is not None and
                        attack_state.get('last_detected', False))
        attack_input = (
            {
                'was_detected':     True,
                'previous_attack':  attack_state['last_attack'],
                'detection_reason': attack_state.get('last_detection_reason', 'pattern detected'),
                'known_rules':      ring_scout.rules,
            }
            if was_detected else
            {'difficulty': difficulty}
        )
        attack = await fraud_gpt.execute(attack_input)

    # Only count after attack is successfully obtained — prevents
    # round counter racing ahead if something fails before this point
    attack_state['round'] += 1
    attack_state['attacks_launched'] += 1
    attack_state['last_attack'] = attack
    round_num = attack_state['round']

    _log(f'--- Round {round_num} | difficulty {difficulty} | '
         f'{"REAL DATA" if REAL_DATA_MODE else "SYNTHETIC"} | '
         f'strategy: {attack.get("strategy","?")} ---', 'info', attack_state)

    _debug('ATTACK LOADED', {
        'strategy':       attack.get('strategy'),
        'n_transactions': len(attack.get('transactions', [])),
        'is_real_fraud':  attack.get('true_fraud', False),
        'sample_amounts': [t.get('amount') for t in attack.get('transactions', [])[:3]],
    })

    # ── 2. ML scoring ─────────────────────────────────────────────────
    scored_txns            = _score_transactions(attack.get('transactions', []))
    attack['transactions'] = scored_txns
    ml_flags  = sum(1 for t in scored_txns if t.get('ml_flagged'))
    avg_score = (sum(t.get('fraud_score', 0) for t in scored_txns) /
                 max(len(scored_txns), 1))

    if ml_flags > 0:
        _log(f'ML Scorer: {ml_flags}/{len(scored_txns)} transactions flagged '
             f'(avg score {avg_score:.0f}/100)', 'info', attack_state)

    # ── 3. Ingest into graph ──────────────────────────────────────────
    await graph_builder.execute(attack)

    for txn in scored_txns:
        for acc_key in ('from', 'to'):
            node_id = txn.get(acc_key)
            if node_id and node_id in graph_builder.graph:
                graph_builder.graph.nodes[node_id]['fraud_score'] = int(txn.get('fraud_score', 0))
                graph_builder.graph.nodes[node_id]['ml_flagged']  = bool(txn.get('ml_flagged', False))

    # ── 4. Ring Scout scans ───────────────────────────────────────────
    rings    = await ring_scout.execute()
    detected = len(rings) > 0
    _update_accuracy(attack, detected, attack_state)

    _debug('RING SCOUT', {
        'rings_found': len(rings),
        'details':     [(r['ring_id'], r['suspicion_score'], r['patterns']) for r in rings]
                       if rings else 'NONE — attack evaded',
        'graph_nodes': graph_builder.graph.number_of_nodes(),
        'graph_edges': graph_builder.graph.number_of_edges(),
    })

    if detected:
        attack_state['detections']           += 1
        attack_state['last_detected']         = True
        ring = rings[0]
        attack_state['last_detection_reason'] = ', '.join(ring.get('patterns', []))

        if REAL_DATA_MODE:
            ring['data_source']      = 'Kaggle Credit Card Fraud Dataset'
            ring['true_fraud_label'] = bool(attack.get('true_fraud', False))

        confirmed_tag = ' [REAL FRAUD CONFIRMED]' if ring.get('true_fraud_label') else ''
        _log(f'Ring Scout flagged {ring["ring_id"]} '
             f'(score {ring["suspicion_score"]}/100 | '
             f'patterns: {", ".join(ring["patterns"])} | '
             f'ML active: {ring.get("ml_active", False)})'
             + confirmed_tag, 'detect', attack_state)

        try:
            report_result         = await investigation.execute(ring)
            ring['cached_report'] = report_result['report']
            _log('Investigation report generated + governance logged.', 'info', attack_state)
        except Exception as e:
            _debug('REPORT ERROR', {'error': str(e), 'type': type(e).__name__})
            ring['cached_report'] = (
                'FRAUD RING DETECTED\n\n'
                f'Ring ID: {ring["ring_id"]}\n'
                f'Suspicion Score: {ring["suspicion_score"]}/100\n'
                f'Patterns: {", ".join(ring.get("patterns", []))}\n'
                f'Amount: ${ring.get("total_amount", 0):,.2f}\n\n'
                '(Full report unavailable — LLM error)'
            )

        return sanitize({
            'round':          round_num,
            'outcome':        'detected',
            'attack':         attack,
            'ring':           ring,
            'report':         ring.get('cached_report', ''),
            'real_data_mode': REAL_DATA_MODE,
            'ml_stats':       {'flagged': ml_flags, 'total': len(scored_txns), 'avg_score': round(avg_score, 1)},
            'graph': graph_builder.to_cytoscape(highlight_accounts=ring['accounts'])
        })

    else:
        attack_state['evasions']     += 1
        attack_state['last_detected'] = False
        fraud_gpt.successful_evasions.append(attack.get('strategy', 'unknown'))

        _log(f'Ring Scout MISSED: {attack["strategy"]} '
             f'(ML flagged {ml_flags}/{len(scored_txns)} — not enough for ring confirmation)',
             'evade', attack_state)

        try:
            adaptation = await defense_ai.execute({
                'attack':         attack,
                'evasion_reason': (
                    f'Strategy "{attack["strategy"]}" evaded all {len(ring_scout.rules)} rules. '
                    f'ML flagged only {ml_flags}/{len(scored_txns)} transactions.'
                )
            })
        except Exception as e:
            _debug('DEFENSEAI ERROR', {'error': str(e)})
            adaptation = {
                'rule_name':      f'adaptive_rule_{defense_ai.evasion_count}',
                'description':    'Auto-generated rule from evasion event',
                'graph_property': 'out_degree',
                'threshold':      'out_degree >= 3',
                'weight':         25,
                'confidence':     60,
            }
            ring_scout.add_rule(adaptation['rule_name'], weight=25)

        attack_state['rules_added'] += 1
        _log(f'DefenseAI: {adaptation["rule_name"]} — {adaptation["description"]}',
             'adapt', attack_state)

        return sanitize({
            'round':          round_num,
            'outcome':        'evaded',
            'attack':         attack,
            'adaptation':     adaptation,
            'real_data_mode': REAL_DATA_MODE,
            'ml_stats':       {'flagged': ml_flags, 'total': len(scored_txns), 'avg_score': round(avg_score, 1)},
            'graph': graph_builder.to_cytoscape()
        })

    # ── Seed legitimate background traffic ────────────────────────────
    if REAL_DATA_MODE and data_streamer:
        legit_txns   = data_streamer.get_legit_batch(n=15)
        scored_legit = _score_transactions(legit_txns)
        for txn in scored_legit:
            graph_builder.add_transaction(txn)
        _log(f'Seeded {len(scored_legit)} real legitimate transactions', 'info')
    else:
        _LEGIT = [f'CUST_{i:03d}' for i in range(1, 21)]
        pairs_used, seeded, attempts = set(), 0, 0
        while seeded < 8 and attempts < 40:
            attempts += 1
            src, dst = random.sample(_LEGIT, 2)
            pair     = tuple(sorted([src, dst]))
            if pair in pairs_used:
                continue
            pairs_used.add(pair)
            graph_builder.add_transaction({
                'from': src, 'to': dst,
                'amount':             round(random.uniform(15, 50), 2),
                'delay_minutes':      0,
                'device_trust_score': random.randint(70, 100),
            })
            seeded += 1

    # ── 1. Get attack ─────────────────────────────────────────────────
    if REAL_DATA_MODE and data_streamer:
        attack = (data_streamer.get_hard_cases(n=5)
                  if difficulty >= 4 else
                  data_streamer.get_fraud_ring(size=6))
        battle_state['attacks_launched'] += 1
        battle_state['last_attack'] = attack
        _log(f'Injecting real fraud: {attack["strategy"]} '
             f'({len(attack["transactions"])} transactions — KAGGLE DATA)', 'attack')
    else:
        was_detected = (battle_state['last_attack'] is not None and
                        battle_state.get('last_detected', False))
        attack_input = (
            {
                'was_detected':     True,
                'previous_attack':  battle_state['last_attack'],
                'detection_reason': battle_state.get('last_detection_reason', 'pattern detected'),
                'known_rules':      ring_scout.rules,
            }
            if was_detected else
            {'difficulty': difficulty}
        )
        attack = await fraud_gpt.execute(attack_input)
        battle_state['attacks_launched'] += 1
        battle_state['last_attack'] = attack
        _log(f'FraudGPT: {attack["strategy"]} '
             f'({len(attack["transactions"])} transactions)', 'attack')

    _debug('ATTACK LOADED', {
        'strategy':       attack.get('strategy'),
        'n_transactions': len(attack.get('transactions', [])),
        'is_real_fraud':  attack.get('true_fraud', False),
        'sample_amounts': [t.get('amount') for t in attack.get('transactions', [])[:3]],
    })

    # ── 2. ML scoring ─────────────────────────────────────────────────
    scored_txns            = _score_transactions(attack.get('transactions', []))
    attack['transactions'] = scored_txns
    ml_flags  = sum(1 for t in scored_txns if t.get('ml_flagged'))
    avg_score = (sum(t.get('fraud_score', 0) for t in scored_txns) /
                 max(len(scored_txns), 1))

    _debug('ML SCORING', {
        'total':     len(scored_txns),
        'flagged':   ml_flags,
        'avg_score': round(avg_score, 1),
        'sample':    [(t.get('from','?'), t.get('fraud_score',0), t.get('ml_flagged',False))
                      for t in scored_txns[:3]],
    })

    if ml_flags > 0:
        _log(f'ML Scorer: {ml_flags}/{len(scored_txns)} transactions flagged '
             f'(avg score {avg_score:.0f}/100)', 'info')

    # ── 3. Ingest into graph ──────────────────────────────────────────
    await graph_builder.execute(attack)

    for txn in scored_txns:
        for acc_key in ('from', 'to'):
            node_id = txn.get(acc_key)
            if node_id and node_id in graph_builder.graph:
                graph_builder.graph.nodes[node_id]['fraud_score'] = int(txn.get('fraud_score', 0))
                graph_builder.graph.nodes[node_id]['ml_flagged']  = bool(txn.get('ml_flagged', False))

    # ── 4. Ring Scout scans ───────────────────────────────────────────
    rings    = await ring_scout.execute()
    detected = len(rings) > 0
    _update_accuracy(attack, detected)

    _debug('RING SCOUT', {
        'rings_found': len(rings),
        'details':     [(r['ring_id'], r['suspicion_score'], r['patterns']) for r in rings]
                       if rings else 'NONE — attack evaded',
        'graph_nodes': graph_builder.graph.number_of_nodes(),
        'graph_edges': graph_builder.graph.number_of_edges(),
    })

    if detected:
        battle_state['detections']           += 1
        battle_state['last_detected']         = True
        ring = rings[0]
        battle_state['last_detection_reason'] = ', '.join(ring.get('patterns', []))

        if REAL_DATA_MODE:
            ring['data_source']      = 'Kaggle Credit Card Fraud Dataset'
            ring['true_fraud_label'] = bool(attack.get('true_fraud', False))

        confirmed_tag = ' [REAL FRAUD CONFIRMED]' if ring.get('true_fraud_label') else ''
        _log(f'Ring Scout flagged {ring["ring_id"]} '
             f'(score {ring["suspicion_score"]}/100 | '
             f'patterns: {", ".join(ring["patterns"])} | '
             f'ML active: {ring.get("ml_active", False)})'
             + confirmed_tag, 'detect')

        try:
            report_result         = await investigation.execute(ring)
            ring['cached_report'] = report_result['report']
            _log('Investigation report generated + governance logged.', 'info')
            _debug('REPORT GENERATED', {
                'ring_id':       ring['ring_id'],
                'report_length': len(ring['cached_report']),
                'preview':       ring['cached_report'][:120],
            })
        except Exception as e:
            _debug('REPORT ERROR', {'error': str(e), 'type': type(e).__name__})
            ring['cached_report'] = (
                'FRAUD RING DETECTED\n\n'
                f'Ring ID: {ring["ring_id"]}\n'
                f'Suspicion Score: {ring["suspicion_score"]}/100\n'
                f'Patterns: {", ".join(ring.get("patterns", []))}\n'
                f'Amount: ${ring.get("total_amount", 0):,.2f}\n\n'
                '(Full report unavailable — LLM error)'
            )

        return sanitize({
            'round':          round_num,
            'outcome':        'detected',
            'attack':         attack,
            'ring':           ring,
            'report':         ring.get('cached_report', ''),
            'real_data_mode': REAL_DATA_MODE,
            'ml_stats': {
                'flagged':   ml_flags,
                'total':     len(scored_txns),
                'avg_score': round(avg_score, 1),
            },
            'graph': graph_builder.to_cytoscape(highlight_accounts=ring['accounts'])
        })

    else:
        battle_state['evasions']     += 1
        battle_state['last_detected'] = False
        fraud_gpt.successful_evasions.append(attack.get('strategy', 'unknown'))

        _log(f'Ring Scout MISSED: {attack["strategy"]} '
             f'(ML flagged {ml_flags}/{len(scored_txns)} — not enough for ring confirmation)',
             'evade')

        try:
            adaptation = await defense_ai.execute({
                'attack':         attack,
                'evasion_reason': (
                    f'Strategy "{attack["strategy"]}" evaded all {len(ring_scout.rules)} rules. '
                    f'ML flagged only {ml_flags}/{len(scored_txns)} transactions.'
                )
            })
        except Exception as e:
            _debug('DEFENSEAI ERROR', {'error': str(e)})
            adaptation = {
                'rule_name':      f'adaptive_rule_{defense_ai.evasion_count}',
                'description':    'Auto-generated rule from evasion event',
                'graph_property': 'out_degree',
                'threshold':      'out_degree >= 3',
                'weight':         25,
                'confidence':     60,
            }
            ring_scout.add_rule(adaptation['rule_name'], weight=25)

        battle_state['rules_added'] += 1
        _log(f'DefenseAI: {adaptation["rule_name"]} — '
             f'{adaptation["description"]} '
             f'[{adaptation.get("graph_property","")} {adaptation.get("threshold","")}]',
             'adapt')

        _debug('EVASION + ADAPTATION', {
            'attack_strategy':  attack.get('strategy'),
            'new_rule':         adaptation.get('rule_name'),
            'rule_description': adaptation.get('description'),
            'total_rules_now':  len(ring_scout.rules),
        })

        return sanitize({
            'round':          round_num,
            'outcome':        'evaded',
            'attack':         attack,
            'adaptation':     adaptation,
            'real_data_mode': REAL_DATA_MODE,
            'ml_stats': {
                'flagged':   ml_flags,
                'total':     len(scored_txns),
                'avg_score': round(avg_score, 1),
            },
            'graph': graph_builder.to_cytoscape()
        })


# ════════════════════════════════════════════════════════════════════════════
# BATTLE MODE (Tab 2) — run_battle_round_ai
# FraudGPT ALWAYS generates the attack via LLM — never DataStreamer.
# DataStreamer only seeds legitimate background traffic.
# Both agents accumulate memory across rounds and learn from each other.
# ════════════════════════════════════════════════════════════════════════════
async def run_battle_round_ai() -> dict:
    """
    One genuine AI-vs-AI battle round.
    — FraudGPT generates attack with full feedback from previous round
    — XGBoost scores every transaction
    — RingScout detects or misses
    — On miss: DefenseAI proposes a new graph rule and injects it live
    — Both agents' memories update so the next round is different
    """
    difficulty = battle_round_state['difficulty']

    graph_builder.reset()

    # ── Seed legitimate background (real data if available) ──────────
    if REAL_DATA_MODE and data_streamer:
        legit_txns   = data_streamer.get_legit_batch(n=15)
        scored_legit = _score_transactions(legit_txns)
        for txn in scored_legit:
            graph_builder.add_transaction(txn)
        _log(f'Seeded {len(scored_legit)} real legitimate transactions as background noise', 'info')
    else:
        _LEGIT = [f'CUST_{i:03d}' for i in range(1, 21)]
        pairs_used, seeded, attempts = set(), 0, 0
        while seeded < 8 and attempts < 40:
            attempts += 1
            src, dst = random.sample(_LEGIT, 2)
            pair     = tuple(sorted([src, dst]))
            if pair in pairs_used:
                continue
            pairs_used.add(pair)
            graph_builder.add_transaction({
                'from': src, 'to': dst,
                'amount':             round(random.uniform(15, 50), 2),
                'delay_minutes':      0,
                'device_trust_score': random.randint(70, 100),
            })
            seeded += 1

    # ── Build FraudGPT input with full feedback from last round ──────
    prev    = battle_round_state['last_attack']
    was_det = battle_round_state['last_detected']

    if prev and was_det:
        attack_input = {
            'was_detected':     True,
            'previous_attack':  prev,
            'detection_reason': battle_round_state['last_detection_reason'],
            'active_rules':     ring_scout.rules,
        }
    elif prev and not was_det:
        attack_input = {
            'was_evaded':      True,
            'previous_attack': prev,
            'difficulty':      difficulty,
            'active_rules':    ring_scout.rules,
        }
    else:
        attack_input = {
            'difficulty':   difficulty,
            'active_rules': ring_scout.rules,
        }

    # ── FraudGPT generates the attack via LLM ────────────────────────
    attack = await battle_fraud_gpt.execute(attack_input)

    # Only count the round once we have a real attack — prevents
    # round counter racing ahead when LLM calls fail mid-function
    battle_state['round'] += 1
    battle_state['attacks_launched'] += 1
    round_num = battle_state['round']
    battle_round_state['last_attack'] = attack

    _log(f'[BATTLE R{round_num}] FraudGPT strategy: "{attack.get("strategy","?")}" '
         f'({len(attack.get("transactions",[]))} txns)', 'attack')

    _debug('BATTLE ATTACK', {
        'round':          round_num,
        'difficulty':     difficulty,
        'strategy':       attack.get('strategy'),
        'n_transactions': len(attack.get('transactions', [])),
        'fg_memory':      str(battle_fraud_gpt.get_memory_state()),
    })

    # ── ML scoring ────────────────────────────────────────────────────
    scored_txns            = _score_transactions(attack.get('transactions', []))
    attack['transactions'] = scored_txns
    ml_flags               = sum(1 for t in scored_txns if t.get('ml_flagged'))
    avg_score              = (sum(t.get('fraud_score', 0) for t in scored_txns) /
                              max(len(scored_txns), 1))

    if ml_flags > 0:
        _log(f'[BATTLE R{round_num}] ML Scorer: {ml_flags}/{len(scored_txns)} flagged '
             f'(avg {avg_score:.0f}/100)', 'info')

    # ── Ingest into graph ─────────────────────────────────────────────
    await graph_builder.execute(attack)
    for txn in scored_txns:
        for acc_key in ('from', 'to'):
            node_id = txn.get(acc_key)
            if node_id and node_id in graph_builder.graph:
                graph_builder.graph.nodes[node_id]['fraud_score'] = int(txn.get('fraud_score', 0))
                graph_builder.graph.nodes[node_id]['ml_flagged']  = bool(txn.get('ml_flagged', False))

    # ── RingScout scans ───────────────────────────────────────────────
    rings    = await ring_scout.execute()
    detected = len(rings) > 0
    _update_accuracy(attack, detected, battle_state)

    if detected:
        battle_state['detections']  += 1
        battle_state['last_detected'] = True
        ring   = rings[0]
        reason = ', '.join(ring.get('patterns', []))

        # Update per-round state
        battle_round_state['last_detected']         = True
        battle_round_state['last_detection_reason'] = reason

        # Write failure into FraudGPT's memory so it adapts next round
        battle_fraud_gpt.failed_attacks.append({
            'strategy': attack.get('strategy', 'unknown'),
            'reason':   reason,
        })

        _log(f'[BATTLE R{round_num}] RingScout: CAUGHT "{attack.get("strategy","?")}" '
             f'(score {ring["suspicion_score"]}/100 | patterns: {reason})', 'detected')

        # Generate investigation report
        try:
            report_result         = await investigation.execute(ring)
            ring['cached_report'] = report_result['report']
        except Exception as e:
            _debug('BATTLE REPORT ERROR', {'error': str(e)})
            ring['cached_report'] = (
                f'Ring {ring["ring_id"]} detected.\n'
                f'Score: {ring["suspicion_score"]}/100\n'
                f'Patterns: {reason}\n'
                f'Amount: ${ring.get("total_amount", 0):,.2f}'
            )

        return sanitize({
            'round':           round_num,
            'difficulty':      difficulty,
            'outcome':         'detected',
            'attack':          attack,
            'ring':            ring,
            'report':          ring.get('cached_report', ''),
            'ml_stats':        {'flagged': ml_flags, 'total': len(scored_txns), 'avg': round(avg_score, 1)},
            'graph':           graph_builder.to_cytoscape(highlight_accounts=ring['accounts']),
            'fraudgpt_memory': battle_fraud_gpt.get_memory_state(),
            'active_rules':    ring_scout.rules,
            'adaptations':     battle_defense_ai.adaptations,
        })

    else:
        battle_state['evasions']      += 1
        battle_state['last_detected']  = False

        # Update per-round state
        battle_round_state['last_detected'] = False

        # Write success into FraudGPT's memory — allow duplicates so counts are accurate
        strat = attack.get('strategy', 'unknown')
        battle_fraud_gpt.successful_evasions.append(strat)

        _log(f'[BATTLE R{round_num}] FraudGPT EVADED — '
             f'ML flagged {ml_flags}/{len(scored_txns)}, not enough for ring confirmation', 'evaded')

        # DefenseAI analyses the evasion and proposes a new rule
        try:
            adaptation = await battle_defense_ai.execute({
                'attack':         attack,
                'evasion_reason': (
                    f'Strategy "{attack.get("strategy","?")}" evaded all '
                    f'{len(ring_scout.rules)} active rules. '
                    f'ML flagged only {ml_flags}/{len(scored_txns)} transactions '
                    f'(avg score {avg_score:.0f}/100). '
                    f'FraudGPT has evaded {len(battle_fraud_gpt.successful_evasions)} time(s) total.'
                ),
            })
        except Exception as e:
            _debug('BATTLE DEFENSEAI ERROR', {'error': str(e)})
            adaptation = {
                'rule_name':      f'battle_rule_{battle_defense_ai.evasion_count}',
                'description':    'Adaptive rule from battle evasion',
                'graph_property': 'out_degree',
                'threshold':      'out_degree >= 3',
                'weight':         25,
                'confidence':     60,
            }
            ring_scout.add_rule(adaptation['rule_name'], weight=25)

        battle_state['rules_added'] += 1
        _log(f'[BATTLE R{round_num}] DefenseAI: new rule "{adaptation["rule_name"]}" — '
             f'{adaptation.get("description","")} '
             f'[{adaptation.get("graph_property","")} {adaptation.get("threshold","")}]', 'rule')

        _debug('BATTLE EVASION + ADAPTATION', {
            'strategy':        attack.get('strategy'),
            'new_rule':        adaptation.get('rule_name'),
            'description':     adaptation.get('description'),
            'total_rules_now': len(ring_scout.rules),
            'fg_wins_total':   len(battle_fraud_gpt.successful_evasions),
        })

        return sanitize({
            'round':           round_num,
            'difficulty':      difficulty,
            'outcome':         'evaded',
            'attack':          attack,
            'adaptation':      adaptation,
            'ml_stats':        {'flagged': ml_flags, 'total': len(scored_txns), 'avg': round(avg_score, 1)},
            'graph':           graph_builder.to_cytoscape(),
            'fraudgpt_memory': battle_fraud_gpt.get_memory_state(),
            'active_rules':    ring_scout.rules,
            'adaptations':     battle_defense_ai.adaptations,
        })


# ── Auto battle loop ──────────────────────────────────────────────────────
def _log_battle_event(result):
    outcome = result.get('outcome', '?')
    rnd     = result.get('round', '?')
    strat   = result.get('attack', {}).get('strategy', '?')
    if outcome == 'detected':
        score = result.get('ring', {}).get('suspicion_score', 0)
        _log(f'[BATTLE R{rnd}] DefenseAI caught "{strat}" — score {score}/100', 'detected')
    else:
        rule = result.get('adaptation', {}).get('rule_name', '?')
        _log(f'[BATTLE R{rnd}] FraudGPT evaded with "{strat}" → new rule: {rule}', 'evaded')


async def _auto_battle():
    """
    Genuine AI vs AI auto-loop.
    FraudGPT always uses the LLM — never DataStreamer.
    Difficulty escalates every 2 rounds up to 5.
    """
    while battle_state['running']:
        try:
            result = await run_battle_round_ai()
            _log_battle_event(result)
        except Exception as e:
            _debug('AUTO BATTLE ERROR', {'error': str(e)})
        # Escalate difficulty every 2 rounds
        battle_round_state['difficulty'] = min((battle_state['round'] // 2) + 1, 5)
        await asyncio.sleep(6)


# ════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get('/')
def root():
    return {
        'status':         'SyndicateAI v2.0 running',
        'round':          battle_state['round'],
        'real_data_mode': REAL_DATA_MODE,
        'ml_trained':     scorer.trained,
        'ml_auc':         scorer.metrics.get('auc_roc'),
    }


@app.post('/reset')
def reset():
    graph_builder.reset()
    noise_state['tx_count'] = 0
    # Reset Attack Mode state
    attack_state.update({
        'round': 0, 'attacks_launched': 0, 'detections': 0, 'evasions': 0,
        'rules_added': 0, 'true_positives': 0, 'false_positives': 0,
        'false_negatives': 0, 'log': [], 'last_attack': None,
        'last_detected': False, 'last_detection_reason': '',
    })
    # Reset Battle Mode state
    battle_state.update({
        'running': False, 'round': 0, 'attacks_launched': 0,
        'detections': 0, 'evasions': 0, 'rules_added': 0,
        'true_positives': 0, 'false_positives': 0, 'false_negatives': 0,
        'log': [], 'last_attack': None,
        'last_detected': False, 'last_detection_reason': '',
    })
    # Reset Attack Mode agent memory
    fraud_gpt.successful_evasions.clear()
    fraud_gpt.failed_attacks.clear()
    # Reset Battle Mode agent memory
    battle_fraud_gpt.successful_evasions.clear()
    battle_fraud_gpt.failed_attacks.clear()
    battle_round_state.update({
        'last_attack': None, 'last_detected': False,
        'last_detection_reason': '', 'difficulty': 1,
    })
    return {'status': 'reset'}


class RoundRequest(BaseModel):
    difficulty: int = 1


@app.post('/round')
async def trigger_round(req: RoundRequest):
    """Attack Mode (Tab 1) — one manual round using real Kaggle data or FraudGPT fallback."""
    _debug('BUTTON CLICKED — Launch Attack', {
        'difficulty':     req.difficulty,
        'round_so_far':   attack_state['round'],
        'real_data_mode': REAL_DATA_MODE,
        'ml_trained':     scorer.trained,
    })
    try:
        result = await run_one_round(req.difficulty)
        return result
    except Exception as e:
        _debug('UNHANDLED ERROR IN TRIGGER_ROUND', {'error': str(e), 'type': type(e).__name__})
        traceback.print_exc()
        return sanitize({
            'round':   attack_state['round'],
            'outcome': 'error',
            'error':   str(e),
            'graph':   graph_builder.to_cytoscape()
        })


@app.post('/battle/round')
async def trigger_battle_round():
    """Battle Mode (Tab 2) — one AI-vs-AI round, always uses FraudGPT LLM."""
    try:
        return await run_battle_round_ai()
    except Exception as e:
        traceback.print_exc()
        return sanitize({'outcome': 'error', 'error': str(e)})


@app.get('/battle/state')
def get_battle_state():
    """Full memory state of both battle agents — used by the battle mode frontend."""
    from agents.ring_scout import RULE_WEIGHTS
    return sanitize({
        'fraudgpt': battle_fraud_gpt.get_memory_state(),
        'defenseai': {
            'evasion_count': battle_defense_ai.evasion_count,
            'adaptations':   battle_defense_ai.adaptations,
        },
        'active_rules':  ring_scout.rules,
        'rule_weights':  dict(RULE_WEIGHTS),
        'round':         battle_state['round'],
        'detections':    battle_state['detections'],
        'evasions':      battle_state['evasions'],
        'difficulty':    battle_round_state['difficulty'],
        'last_detected': battle_round_state['last_detected'],
    })


@app.get('/stats')
def get_stats():
    # Attack Mode accuracy metrics
    atp = attack_state['true_positives']
    afp = attack_state['false_positives']
    afn = attack_state['false_negatives']
    a_precision = round(atp / (atp + afp), 3) if (atp + afp) > 0 else None
    a_recall    = round(atp / (atp + afn), 3) if (atp + afn) > 0 else None
    a_f1        = (round(2 * a_precision * a_recall / (a_precision + a_recall), 3)
                   if a_precision and a_recall else None)

    # Battle Mode accuracy metrics
    btp = battle_state['true_positives']
    bfp = battle_state['false_positives']
    bfn = battle_state['false_negatives']
    b_precision = round(btp / (btp + bfp), 3) if (btp + bfp) > 0 else None
    b_recall    = round(btp / (btp + bfn), 3) if (btp + bfn) > 0 else None
    b_f1        = (round(2 * b_precision * b_recall / (b_precision + b_recall), 3)
                   if b_precision and b_recall else None)

    # Rules created by Battle Mode's DefenseAI specifically
    battle_rule_names = {a['rule_name'] for a in battle_defense_ai.adaptations}

    return sanitize({
        # ── Attack Mode stats (Tab 1) ──────────────────────────────────
        'attack': {
            'round':            attack_state['round'],
            'attacks_launched': attack_state['attacks_launched'],
            'detections':       attack_state['detections'],
            'evasions':         attack_state['evasions'],
            'rules_added':      attack_state['rules_added'],
            'precision':        a_precision,
            'recall':           a_recall,
            'f1':               a_f1,
            'adaptations':      defense_ai.adaptations,
            'fraud_gpt_memory': fraud_gpt.get_memory_state(),
            'log':              attack_state['log'],
        },
        # ── Battle Mode stats (Tab 2) ──────────────────────────────────
        'battle': {
            'running':          battle_state['running'],
            'round':            battle_state['round'],
            'attacks_launched': battle_state['attacks_launched'],
            'detections':       battle_state['detections'],
            'evasions':         battle_state['evasions'],
            'rules_added':      battle_state['rules_added'],
            'precision':        b_precision,
            'recall':           b_recall,
            'f1':               b_f1,
            'log':              battle_state['log'],
        },
        # ── Shared ────────────────────────────────────────────────────
        'active_rules':       ring_scout.rules,
        'battle_rule_names':  list(battle_rule_names),   # rules earned from battle training
        'adaptations':        defense_ai.adaptations,    # attack mode adaptations
        'real_data_mode':     REAL_DATA_MODE,
        'dataset_stats':      data_streamer.get_stats() if REAL_DATA_MODE else None,
        'ml_model_trained':   scorer.trained,
        'model_metrics': {
            'auc_roc':    scorer.metrics.get('auc_roc'),
            'precision':  scorer.metrics.get('precision'),
            'recall':     scorer.metrics.get('recall'),
            'f1':         scorer.metrics.get('f1'),
            'model_type': scorer.metrics.get('model_type', 'XGBoost'),
        },
        'noise_tx_count':   noise_state['tx_count'],
    })


@app.get('/graph')
def get_graph():
    return sanitize(graph_builder.to_cytoscape())


@app.get('/ring/{ring_id}')
def get_ring(ring_id: str):
    ring = next((r for r in ring_scout.flagged_rings
                 if r['ring_id'] == ring_id), None)
    if not ring:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={'error': 'Ring not found'})
    return sanitize(ring)


@app.get('/log')
def get_log():
    return {'log': battle_state['log']}


@app.get('/governance')
def get_governance():
    entries = GOVERNANCE_LOG
    return sanitize({
        'total_entries': len(entries),
        'entries':       entries,
        'summary': {
            'total_flagged':       len(entries),
            'high_confidence':     sum(1 for e in entries if (e.get('suspicion_score') or 0) >= 80),
            'human_review_needed': sum(1 for e in entries if e.get('human_review_required')),
            'ml_assisted':         sum(1 for e in entries if e.get('ml_active')),
        }
    })


@app.get('/stream/battle')
async def stream_battle():
    async def gen():
        seen = 0
        while True:
            entries = battle_state['log']
            if len(entries) > seen:
                for entry in entries[seen:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                seen = len(entries)
            await asyncio.sleep(0.3)
    return StreamingResponse(gen(), media_type='text/event-stream')


@app.get('/stream/report/{ring_id}')
async def stream_report(ring_id: str):
    ring = next((r for r in ring_scout.flagged_rings
                 if r['ring_id'] == ring_id), None)
    if not ring:
        return {'error': 'Ring not found'}

    async def gen():
        cached = ring.get('cached_report', '')
        if not cached:
            score    = ring.get('suspicion_score', 0)
            patterns = ', '.join(ring.get('patterns', []))
            amount   = ring.get('total_amount', 0)
            cached   = (
                'FRAUD RING DETECTED\n\n'
                'Ring ID: ' + ring_id + '\n'
                'Suspicion Score: ' + str(score) + '/100\n'
                'Patterns: ' + patterns + '\n'
                'Amount: $' + f'{amount:,.2f}'
            )
        chunk_size = 4
        for i in range(0, len(cached), chunk_size):
            yield 'data: ' + json.dumps({'chunk': cached[i:i+chunk_size]}) + '\n\n'
            await asyncio.sleep(0.01)
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.post('/train')
async def train_model():
    if not REAL_DATA_MODE:
        return {'error': 'Real data mode not available — place creditcard.csv in data/'}
    loop    = asyncio.get_event_loop()
    metrics = await loop.run_in_executor(
        None, lambda: scorer.train('data/creditcard.csv')
    )
    return {'status': 'trained', 'metrics': metrics}


@app.get('/model/metrics')
def model_metrics():
    return scorer.metrics if scorer.trained else {'status': 'not trained'}


@app.post('/battle/start')
async def start_battle():
    if battle_state['running']:
        return {'status': 'already running'}
    battle_state['running'] = True
    asyncio.create_task(_auto_battle())
    return {'status': 'battle started'}


@app.post('/battle/stop')
def stop_battle():
    battle_state['running'] = False
    return {'status': 'battle stopped'}


@app.post('/noise/start')
async def start_noise():
    if noise_state['running']:
        return {'status': 'already running'}
    noise_state['running'] = True
    asyncio.create_task(_noise_loop())
    return {'status': 'noise started'}


@app.post('/noise/stop')
def stop_noise():
    noise_state['running'] = False
    return {'status': 'noise stopped'}


@app.on_event('startup')
async def startup():
    noise_state['running'] = True
    loop = asyncio.get_event_loop()
    loop.create_task(_noise_loop())
    print('[Main] Noise loop started.')
    if scorer.trained:
        print(f'[Main] ML model ready — AUC: {scorer.metrics.get("auc_roc")}')
    else:
        print('[Main] ML model not trained — run: python train_model.py')


@app.on_event('shutdown')
async def shutdown():
    noise_state['running']  = False
    battle_state['running'] = False


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False,
                timeout_graceful_shutdown=3)