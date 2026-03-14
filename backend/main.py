"""
SyndicateAI — main FastAPI application.

Three-layer fraud detection:
  1. TransactionScorer (ML)  — per-transaction XGBoost fraud probability
  2. GraphBuilder            — live transaction network with centrality metrics
  3. RingScout               — graph topology + ML consensus ring detection

Red team:
  FraudGPT   — generates adversarial attacks (real data or synthetic)
  DefenseAI  — proposes new rules when attacks evade detection

Governance:
  InvestigationAgent logs every flagged ring to GOVERNANCE_LOG in
  watsonx.governance-compatible format.
"""
import sys
import asyncio
import json
import random
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append('.')

from agents.graph_builder      import GraphBuilder
from agents.ring_scout         import RingScout
from agents.investigation_agent import InvestigationAgent, GOVERNANCE_LOG
from agents.fraud_gpt          import FraudGPT
from agents.defense_ai         import DefenseAI
from agents.transaction_scorer import TransactionScorer
from data_streamer             import DataStreamer

# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(title='SyndicateAI', version='2.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*']
)

# ── Agents ───────────────────────────────────────────────────────────────
graph_builder = GraphBuilder()
ring_scout    = RingScout(graph_builder)
investigation = InvestigationAgent()
fraud_gpt     = FraudGPT()
defense_ai    = DefenseAI(ring_scout)
scorer        = TransactionScorer()

# ── Real data ────────────────────────────────────────────────────────────
try:
    data_streamer  = DataStreamer('data/creditcard.csv')
    REAL_DATA_MODE = True
    print('[Main] Real data mode ENABLED')
except Exception as e:
    data_streamer  = None
    REAL_DATA_MODE = False
    print(f'[Main] Real data mode DISABLED ({e}) — using synthetic data')

# ── Battle state ─────────────────────────────────────────────────────────
battle_state = {
    'running':               False,
    'round':                 0,
    'attacks_launched':      0,
    'detections':            0,
    'evasions':              0,
    'rules_added':           0,
    'true_positives':        0,
    'false_positives':       0,
    'false_negatives':       0,
    'log':                   [],
    'last_attack':           None,
    'last_detected':         False,
    'last_detection_reason': '',
}

_LEGIT_ACCOUNTS = [f'CUST_{i:03d}' for i in range(1, 21)]
noise_state     = {'running': False, 'tx_count': 0}


# ── Background noise ─────────────────────────────────────────────────────
async def _noise_loop():
    while noise_state['running']:
        for _ in range(random.randint(1, 3)):
            src, dst = random.sample(_LEGIT_ACCOUNTS, 2)
            graph_builder.add_transaction({
                'from': src, 'to': dst,
                'amount':        round(random.uniform(15, 50), 2),
                'delay_minutes': 0,
                'ip':            f'192.168.{random.randint(1,10)}.{random.randint(1,255)}',
                'device':        f'device_{random.randint(1, 15):02d}',
                'device_trust_score':  random.randint(70, 100),
                'location_mismatch':   0,
                'foreign_transaction': 0,
                'velocity_last_24h':   random.randint(1, 3),
            })
            noise_state['tx_count'] += 1
        await asyncio.sleep(2)


# ── Helpers ──────────────────────────────────────────────────────────────
def _log(message: str, kind: str = 'info'):
    entry = {
        'time':    datetime.utcnow().strftime('%H:%M:%S'),
        'message': message,
        'kind':    kind,
    }
    battle_state['log'].append(entry)
    print(f"[BATTLE] {entry['time']} [{kind.upper()}] {message}")
    return entry


def _update_accuracy(attack: dict, detected: bool):
    if not REAL_DATA_MODE:
        return
    is_fraud = attack.get('true_fraud', False)
    if   is_fraud and     detected: battle_state['true_positives']  += 1
    elif is_fraud and not detected: battle_state['false_negatives'] += 1
    elif not is_fraud and detected: battle_state['false_positives'] += 1


def _score_transactions(transactions: list) -> list:
    """
    Run ML scorer on transactions.
    Adds fraud_score and ml_flagged to each transaction dict.
    Falls back to heuristic if model not trained.
    """
    if scorer.trained:
        return scorer.score_batch(transactions)
    # Heuristic fallback — still annotates each transaction
    return [scorer.score_transaction(t) for t in transactions]


# ── Core battle round ────────────────────────────────────────────────────
async def run_one_round(difficulty: int = 1) -> dict:
    battle_state['round'] += 1
    round_num = battle_state['round']
    _log(f'--- Round {round_num} | difficulty {difficulty} | '
         f'{"REAL DATA" if REAL_DATA_MODE else "SYNTHETIC"} ---', 'info')

    graph_builder.reset()

    # ── Seed background legitimate traffic ────────────────────────────
    if REAL_DATA_MODE and data_streamer:
        legit_txns = data_streamer.get_legit_batch(n=15)
        # Score them too — they should come back clean
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
                'amount': round(random.uniform(15, 50), 2),
                'delay_minutes': 0,
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
        _log(
            f'Injecting real fraud: {attack["strategy"]} '
            f'({len(attack["transactions"])} transactions — KAGGLE DATA)',
            'attack'
        )
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
        _log(
            f'FraudGPT: {attack["strategy"]} '
            f'({len(attack["transactions"])} transactions)',
            'attack'
        )

    # ── 2. ML scoring ─────────────────────────────────────────────────
    scored_txns = _score_transactions(attack.get('transactions', []))
    attack['transactions'] = scored_txns

    ml_flags = sum(1 for t in scored_txns if t.get('ml_flagged'))
    avg_score = (sum(t.get('fraud_score', 0) for t in scored_txns) /
                 max(len(scored_txns), 1))

    if ml_flags > 0:
        _log(
            f'ML Scorer: {ml_flags}/{len(scored_txns)} transactions flagged '
            f'(avg score {avg_score:.0f}/100)',
            'info'
        )

    # ── 3. Ingest into graph ──────────────────────────────────────────
    await graph_builder.execute(attack)

    # ── 4. Ring Scout scans ───────────────────────────────────────────
    rings    = await ring_scout.execute()
    detected = len(rings) > 0
    _update_accuracy(attack, detected)

    if detected:
        battle_state['detections']           += 1
        battle_state['last_detected']         = True
        ring = rings[0]
        battle_state['last_detection_reason'] = ', '.join(ring.get('patterns', []))

        # Annotate with data context
        if REAL_DATA_MODE:
            ring['data_source']      = 'Kaggle Credit Card Fraud Dataset'
            ring['true_fraud_label'] = attack.get('true_fraud', False)

        confirmed_tag = ' [REAL FRAUD CONFIRMED]' if ring.get('true_fraud_label') else ''
        _log(
            f'Ring Scout flagged {ring["ring_id"]} '
            f'(score {ring["suspicion_score"]}/100 | '
            f'patterns: {", ".join(ring["patterns"])} | '
            f'ML active: {ring.get("ml_active", False)})'
            + confirmed_tag,
            'detect'
        )

        report_result = await investigation.execute(ring)
        _log('Investigation report generated + governance logged.', 'info')

        return {
            'round':          round_num,
            'outcome':        'detected',
            'attack':         attack,
            'ring':           ring,
            'report':         report_result['report'],
            'real_data_mode': REAL_DATA_MODE,
            'ml_stats': {
                'flagged':   ml_flags,
                'total':     len(scored_txns),
                'avg_score': round(avg_score, 1),
            },
            'graph': graph_builder.to_cytoscape(highlight_accounts=ring['accounts'])
        }

    else:
        battle_state['evasions']     += 1
        battle_state['last_detected'] = False

        # Record successful evasion in FraudGPT memory
        fraud_gpt.successful_evasions.append(attack)

        _log(
            f'Ring Scout MISSED: {attack["strategy"]} '
            f'(ML flagged {ml_flags}/{len(scored_txns)} — not enough for ring confirmation)',
            'evade'
        )

        adaptation = await defense_ai.execute({
            'attack':         attack,
            'evasion_reason': (
                f'Strategy "{attack["strategy"]}" evaded all {len(ring_scout.rules)} rules. '
                f'ML flagged only {ml_flags}/{len(scored_txns)} transactions.'
            )
        })
        battle_state['rules_added'] += 1
        _log(
            f'DefenseAI: {adaptation["rule_name"]} — '
            f'{adaptation["description"]} '
            f'[{adaptation.get("graph_property","")} {adaptation.get("threshold","")}]',
            'adapt'
        )

        return {
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
        }


# ── REST endpoints ───────────────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'status':          'SyndicateAI v2.0 running',
        'round':           battle_state['round'],
        'real_data_mode':  REAL_DATA_MODE,
        'ml_trained':      scorer.trained,
        'ml_auc':          scorer.metrics.get('auc_roc'),
    }


@app.post('/reset')
def reset():
    graph_builder.reset()
    noise_state['tx_count'] = 0
    battle_state.update({
        'running': False, 'round': 0,
        'attacks_launched': 0, 'detections': 0, 'evasions': 0, 'rules_added': 0,
        'true_positives': 0, 'false_positives': 0, 'false_negatives': 0,
        'log': [], 'last_attack': None,
        'last_detected': False, 'last_detection_reason': '',
    })
    fraud_gpt.successful_evasions.clear()
    fraud_gpt.attack_history.clear()
    fraud_gpt.failed_attacks.clear()
    return {'status': 'reset'}


class RoundRequest(BaseModel):
    difficulty: int = 1


@app.post('/round')
async def trigger_round(req: RoundRequest):
    return await run_one_round(req.difficulty)


@app.get('/stats')
def get_stats():
    tp = battle_state['true_positives']
    fp = battle_state['false_positives']
    fn = battle_state['false_negatives']
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else None
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else None
    f1        = (round(2 * precision * recall / (precision + recall), 3)
                 if precision and recall else None)

    return {
        **battle_state,
        'graph':           graph_builder.get_stats(),
        'active_rules':    ring_scout.rules,
        'adaptations':     defense_ai.adaptations,
        'fraud_gpt_memory':fraud_gpt.get_memory_state(),
        'noise_tx_count':  noise_state['tx_count'],
        'noise_running':   noise_state['running'],
        'real_data_mode':  REAL_DATA_MODE,
        'dataset_stats':   data_streamer.get_stats() if REAL_DATA_MODE else None,
        'precision':       precision,
        'recall':          recall,
        'f1':              f1,
        'ml_model': {
            'trained':    scorer.trained,
            'auc_roc':    scorer.metrics.get('auc_roc'),
            'precision':  scorer.metrics.get('precision'),
            'recall':     scorer.metrics.get('recall'),
            'f1':         scorer.metrics.get('f1'),
            'model_type': scorer.metrics.get('model_type', 'unknown'),
        },
    }


@app.get('/graph')
def get_graph():
    return graph_builder.to_cytoscape()


@app.get('/log')
def get_log():
    return {'log': battle_state['log']}


@app.get('/governance')
def get_governance():
    return {'entries': GOVERNANCE_LOG, 'total': len(GOVERNANCE_LOG)}


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
        return {'error': f'Ring {ring_id} not found'}

    async def gen():
        async for chunk in investigation.stream_report(ring):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.post('/train')
async def train_model():
    """Train the ML scorer. Call this once after startup."""
    if not REAL_DATA_MODE:
        return {'error': 'Real data mode not available — check CSV path'}
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
    asyncio.create_task(_noise_loop())
    print('[Main] Noise loop started.')
    if scorer.trained:
        print(f'[Main] ML model ready — AUC: {scorer.metrics.get("auc_roc")}')
    else:
        print('[Main] ML model not trained — POST /train to train.')


@app.on_event('shutdown')
async def shutdown():
    noise_state['running']  = False
    battle_state['running'] = False


async def _auto_battle():
    difficulty = 1
    while battle_state['running']:
        await run_one_round(difficulty)
        if battle_state['round'] % 2 == 0:
            difficulty = min(difficulty + 1, 5)
        await asyncio.sleep(5)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)