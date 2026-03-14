import asyncio
from agents.investigation_agent import InvestigationAgent
from agents.fraud_gpt import FraudGPT

async def test():
    print('\n' + '='*60)
    print('TESTING AGENTS')
    print('='*60 + '\n')
    
    print('Testing Investigation Agent...')
    inv = InvestigationAgent()
    report = await inv.execute({
        'ring_id': 'R_001',
        'accounts': ['A1', 'A2', 'A3'],
        'suspicion_score': 85,
        'patterns': ['fan_out', 'structuring'],
        'total_amount': 12000,
        'timeframe_hours': 3
    })
    print(f'\nReport:\n{report["report"]}\n')
    
    print('\n' + '-'*60)
    print('Testing FraudGPT...')
    fraud = FraudGPT()
    attack = await fraud.execute({'difficulty': 1})
    print(f'\nAttack: {attack["strategy"]}')
    print(f'Transactions: {len(attack["transactions"])}')
    
    print('\n' + '-'*60)
    print('Testing Adaptive Attack...')
    attack2 = await fraud.execute({
        'was_detected': True,
        'previous_attack': attack,
        'detection_reason': 'Fan-out pattern detected'
    })
    print(f'\nAdapted: {attack2["strategy"]}')
    
    print('\n' + '='*60)
    print('✅ TESTS COMPLETE')
    print('='*60)

asyncio.run(test())
