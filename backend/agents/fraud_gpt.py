import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient

class FraudGPT(BaseAgent):
    def __init__(self):
        super().__init__('FraudGPT')
        self.llm = LLMClient()
        self.attack_count = 0
    
    async def execute(self, input_data=None):
        self.attack_count += 1
        
        if input_data and input_data.get('was_detected'):
            return await self._adapt_attack(input_data)
        else:
            difficulty = input_data.get('difficulty', 1) if input_data else 1
            return await self._generate_attack(difficulty)
    
    async def _generate_attack(self, difficulty):
        strategies = {
            1: 'fan-out: send from 1 source to 5 accounts',
            2: 'random amounts under $500 each',
            3: 'circular: A->B->C->D->A with delays',
            4: 'layered: multiple hops with noise',
            5: 'geographic spread with time delays'
        }
        
        strategy_desc = strategies.get(difficulty, strategies[1])
        
        prompt = f'''Generate a fraud attack: {strategy_desc}

Move $10,000 total. Output ONLY this JSON:
{{
    "strategy": "brief strategy name",
    "transactions": [
        {{"from": "ACC_ID", "to": "ACC_ID", "amount": 0, "delay_minutes": 0}}
    ]
}}'''

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You are a fraud simulator. Output only valid JSON.',
            max_tokens=400
        )
        
        if 'error' in result:
            return self._fallback_attack(difficulty)
        
        return {
            'attack_id': f'ATK_{self.attack_count}',
            'strategy': result.get('strategy', 'Unknown'),
            'transactions': result.get('transactions', []),
            'difficulty': difficulty
        }
    
    async def _adapt_attack(self, input_data):
        previous = input_data.get('previous_attack', {})
        reason = input_data.get('detection_reason', 'pattern detected')
        
        prompt = f'''Previous attack FAILED: {previous.get('strategy')}
Why: {reason}

Generate a NEW attack that avoids this. Move $10K. Output ONLY JSON:
{{
    "strategy": "new approach",
    "transactions": [...]
}}'''

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You learn from failures. Output only JSON.',
            max_tokens=400
        )
        
        if 'error' in result:
            return self._fallback_attack(2)
        
        return {
            'attack_id': f'ATK_{self.attack_count}_ADAPTED',
            'strategy': result.get('strategy', 'Adapted'),
            'transactions': result.get('transactions', []),
            'is_adaptive': True
        }
    
    def _fallback_attack(self, difficulty):
        return {
            'attack_id': f'ATK_{self.attack_count}_FALLBACK',
            'strategy': 'Fan-out structuring (fallback)',
            'transactions': [
                {'from': 'SOURCE', 'to': 'MULE_1', 'amount': 2000, 'delay_minutes': 0},
                {'from': 'SOURCE', 'to': 'MULE_2', 'amount': 2000, 'delay_minutes': 5},
                {'from': 'SOURCE', 'to': 'MULE_3', 'amount': 2000, 'delay_minutes': 10},
                {'from': 'SOURCE', 'to': 'MULE_4', 'amount': 2000, 'delay_minutes': 15},
                {'from': 'SOURCE', 'to': 'MULE_5', 'amount': 2000, 'delay_minutes': 20}
            ],
            'difficulty': difficulty
        }
