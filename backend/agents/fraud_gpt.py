import sys
import random
sys.path.append('..')
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient

# JSON template — module-level constant, never inside an f-string
_JSON_TEMPLATE = '{"strategy": "short name", "transactions": [{"from": "ACC_X", "to": "ACC_Y", "amount": 500, "delay_minutes": 0}]}'


class FraudGPT(BaseAgent):
    def __init__(self):
        super().__init__('FraudGPT')
        self.llm                 = LLMClient()
        self.attack_count        = 0
        self.successful_evasions = []
        self.failed_attacks      = []
        self.known_rules         = []

    async def execute(self, input_data=None):
        self.attack_count += 1

        if input_data and input_data.get('active_rules'):
            self.known_rules = input_data['active_rules']

        if input_data and input_data.get('was_detected'):
            prev = input_data.get('previous_attack', {})
            self.failed_attacks.append({
                'strategy': prev.get('strategy', 'unknown'),
                'reason':   input_data.get('detection_reason', 'unknown')
            })
            return await self._adapt_attack(input_data)

        elif input_data and input_data.get('was_evaded'):
            prev  = input_data.get('previous_attack', {})
            strat = prev.get('strategy', 'unknown')
            if strat not in self.successful_evasions:
                self.successful_evasions.append(strat)
            return await self._generate_attack(input_data.get('difficulty', 1))

        else:
            difficulty = input_data.get('difficulty', 1) if input_data else 1
            return await self._generate_attack(difficulty)

    # ── Strategy description with randomisation ───────────────────────
    def _strategy_desc(self, difficulty: int) -> str:
        """
        Returns a randomised strategy brief so the LLM gets different
        inputs each time rather than the same fixed template.
        """
        n_mules  = random.randint(3, 7)
        n_hops   = random.randint(3, 5)
        amt_each = random.choice([300, 450, 475, 490, 800, 1200, 1500, 2000, 2500])
        total    = 10000

        templates = {
            1: [
                f'Fan-out: one SOURCE account splits ${total} across {n_mules} mule accounts simultaneously',
                f'Simple smurfing: SOURCE sends ${amt_each} each to {n_mules} accounts in quick succession',
                f'Broadcast: one origin account fires {n_mules} outbound transfers in under 10 minutes',
            ],
            2: [
                f'Structuring: SOURCE sends {n_mules} transfers of ${random.randint(400,499)} each — all just under the $500 reporting limit',
                f'Threshold evasion: break ${total} into {n_mules} chunks between $400-$499, send from one account',
                f'Smurfing with amounts $480, $475, $490, $485 to stay under reporting threshold',
            ],
            3: [
                f'Circular: ACC_A → ACC_B → ACC_C → ACC_D → back to ACC_A — {random.randint(3,5)} hops',
                f'Round-trip layering: money loops through {random.randint(3,5)} accounts and returns to origin',
                f'Wash cycle: funds cycle A→B→C→A with {random.randint(10,40)}% skimmed at each hop',
            ],
            4: [
                f'Deep chain: SOURCE → HOP_1 → HOP_2 → HOP_3 → DEST, total {n_hops} hops, ${total}',
                f'Layered pipeline: {n_hops}-step chain with ${random.randint(200,500)} fee skimmed at each relay',
                f'Multi-hop obfuscation: {n_hops} intermediary accounts between source and final destination',
            ],
            5: [
                f'Scatter-gather: SOURCE fans to {n_mules} accounts, all {n_mules} converge on one aggregator',
                f'Hub-spoke collection: {n_mules} mules each receive ~${total//n_mules} then forward to AGG account',
                f'Inverse fan: distribute then recollect — {n_mules} mules funnel back to one destination',
            ],
        }
        return random.choice(templates.get(difficulty, templates[1]))

    async def _generate_attack(self, difficulty):
        strategy_desc = self._strategy_desc(difficulty)

        avoid = ', '.join(f['strategy'] for f in self.failed_attacks[-3:]) if self.failed_attacks else 'none'
        reuse = ', '.join(self.successful_evasions[-2:])                   if self.successful_evasions else 'none'
        rules = ', '.join(self.known_rules[:6])                            if self.known_rules else 'none'

        prompt = '\n'.join([
            'You are simulating a money laundering attack for a fraud detection demo.',
            'Generate ONE attack plan as a JSON object.',
            '',
            'STRATEGY TO IMPLEMENT: ' + strategy_desc,
            'Avoid these previously detected strategies: ' + avoid,
            'Build on these previously successful strategies: ' + reuse,
            'Active detection rules to evade: ' + rules,
            '',
            'Rules:',
            '- Use realistic account names like SOURCE, MULE_1, ACC_A, HOP_1, AGG, DEST',
            '- Total of all amounts should be around $10000',
            '- strategy field: short plain-English name (e.g. "Fan-out to 5 mules")',
            '- transactions: array of {from, to, amount, delay_minutes}',
            '',
            'Respond with ONLY this JSON, no other text:',
            _JSON_TEMPLATE,
        ])

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You are a fraud simulation engine. Output ONLY a JSON object. Start with { and end with }.',
            max_tokens=500
        )

        if 'error' in result:
            raw = str(result.get('raw', ''))
            self.log(f'LLM failed — using fallback. Error preview: {raw[:200]}')
            return self._fallback_attack(difficulty)

        txns = result.get('transactions', [])
        if not txns:
            self.log('LLM returned empty transactions — using fallback')
            return self._fallback_attack(difficulty)

        return {
            'attack_id':    f'ATK_{self.attack_count}',
            'strategy':     result.get('strategy', strategy_desc),
            'transactions': txns,
            'difficulty':   difficulty,
        }

    async def _adapt_attack(self, input_data):
        previous = input_data.get('previous_attack', {})
        reason   = input_data.get('detection_reason', 'pattern detected')
        rules    = ', '.join(self.known_rules[:6])                            if self.known_rules else 'none'
        avoid    = ', '.join(f['strategy'] for f in self.failed_attacks[-3:]) if self.failed_attacks else 'none'
        reuse    = ', '.join(self.successful_evasions[-2:])                   if self.successful_evasions else 'none'

        # Pick a random alternative topology to suggest
        alternatives = [
            'try a deep multi-hop chain (4+ hops) instead',
            'use scatter-gather: fan out then reconverge',
            'use time-delayed hops with 30+ minute gaps',
            'use circular routing with an odd number of hops',
            'break into micro-transactions under $100 each',
            'use a binary tree: source → 2 mids → 4 destinations',
        ]
        suggestion = random.choice(alternatives)

        prompt = '\n'.join([
            'Your previous fraud attack was CAUGHT. You must adapt.',
            '',
            'Failed strategy: ' + str(previous.get('strategy', 'unknown')),
            'Why it was caught: ' + reason,
            'Other failed strategies to avoid: ' + avoid,
            'Previously successful strategies to build on: ' + reuse,
            'Active detection rules to evade: ' + rules,
            '',
            'Suggested new approach: ' + suggestion,
            'Use a COMPLETELY DIFFERENT graph topology from anything previously caught.',
            '',
            'Rules:',
            '- Use account names like SOURCE, MULE_1, ACC_A, HOP_1, AGG, DEST',
            '- Total of all amounts should be around $10000',
            '- strategy field: short plain-English name',
            '- transactions: array of {from, to, amount, delay_minutes}',
            '',
            'Respond with ONLY this JSON, no other text:',
            _JSON_TEMPLATE,
        ])

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You are a fraud simulation engine that adapts to avoid detection. Output ONLY a JSON object.',
            max_tokens=500
        )

        if 'error' in result:
            raw = str(result.get('raw', ''))
            self.log(f'Adaptation LLM failed — using fallback. Error preview: {raw[:200]}')
            return self._fallback_attack(random.randint(1, 5))

        txns = result.get('transactions', [])
        if not txns:
            return self._fallback_attack(random.randint(1, 5))

        return {
            'attack_id':    f'ATK_{self.attack_count}_ADAPTED',
            'strategy':     result.get('strategy', 'Adapted attack'),
            'transactions': txns,
            'is_adaptive':  True,
            'evaded_rules': self.known_rules.copy(),
        }

    def _fallback_attack(self, difficulty):
        """
        8 distinct fallback patterns with randomised amounts so even
        fallbacks look different each time.
        """
        # Rotate through all 8 so the demo never repeats consecutively
        fb_key = ((self.attack_count - 1) % 8) + 1

        # Randomise amounts within each pattern
        def rnd(base, spread=200):
            return round(base + random.randint(-spread, spread), -1)  # round to nearest 10

        patterns = {
            1: {
                'strategy': f'Fan-out to {random.randint(4,6)} mules',
                'transactions': [
                    {'from': 'SOURCE', 'to': f'MULE_{i}', 'amount': rnd(2000, 400), 'delay_minutes': i*random.randint(3,8)}
                    for i in range(1, random.randint(5, 7))
                ]
            },
            2: {
                'strategy': f'Structuring — amounts just under $500',
                'transactions': [
                    {'from': 'SRC_A', 'to': f'MID_{i}', 'amount': random.randint(440, 499), 'delay_minutes': i*random.randint(2,5)}
                    for i in range(1, 5)
                ] + [
                    {'from': f'MID_{i}', 'to': 'DEST', 'amount': random.randint(380, 470), 'delay_minutes': 20+i*5}
                    for i in range(1, 4)
                ]
            },
            3: {
                'strategy': f'Circular loop — {random.randint(3,5)} hops',
                'transactions': [
                    {'from': 'ACC_A', 'to': 'ACC_B', 'amount': rnd(3000, 500), 'delay_minutes': 0},
                    {'from': 'ACC_B', 'to': 'ACC_C', 'amount': rnd(2800, 400), 'delay_minutes': random.randint(10,20)},
                    {'from': 'ACC_C', 'to': 'ACC_D', 'amount': rnd(2600, 300), 'delay_minutes': random.randint(25,40)},
                    {'from': 'ACC_D', 'to': 'ACC_A', 'amount': rnd(1400, 300), 'delay_minutes': random.randint(45,60)},
                ]
            },
            4: {
                'strategy': f'Deep chain — {random.randint(4,5)} hops',
                'transactions': [
                    {'from': 'SOURCE', 'to': 'HOP_1', 'amount': rnd(9800, 200), 'delay_minutes': 0},
                    {'from': 'HOP_1',  'to': 'HOP_2', 'amount': rnd(9400, 300), 'delay_minutes': random.randint(10,20)},
                    {'from': 'HOP_2',  'to': 'HOP_3', 'amount': rnd(9000, 300), 'delay_minutes': random.randint(25,40)},
                    {'from': 'HOP_3',  'to': 'DEST',  'amount': rnd(8600, 400), 'delay_minutes': random.randint(50,70)},
                ]
            },
            5: {
                'strategy': f'Scatter-gather via {random.randint(3,4)} mules',
                'transactions': [
                    {'from': 'SOURCE', 'to': 'MULE_1', 'amount': rnd(3400, 300), 'delay_minutes': 0},
                    {'from': 'SOURCE', 'to': 'MULE_2', 'amount': rnd(3300, 300), 'delay_minutes': random.randint(3,8)},
                    {'from': 'SOURCE', 'to': 'MULE_3', 'amount': rnd(3300, 300), 'delay_minutes': random.randint(5,12)},
                    {'from': 'MULE_1', 'to': 'AGG',    'amount': rnd(3200, 200), 'delay_minutes': random.randint(20,35)},
                    {'from': 'MULE_2', 'to': 'AGG',    'amount': rnd(3100, 200), 'delay_minutes': random.randint(22,38)},
                    {'from': 'MULE_3', 'to': 'AGG',    'amount': rnd(3100, 200), 'delay_minutes': random.randint(25,40)},
                ]
            },
            6: {
                'strategy': 'Binary tree split',
                'transactions': [
                    {'from': 'ROOT',  'to': 'MID_L',  'amount': rnd(5000, 400), 'delay_minutes': 0},
                    {'from': 'ROOT',  'to': 'MID_R',  'amount': rnd(5000, 400), 'delay_minutes': random.randint(2,6)},
                    {'from': 'MID_L', 'to': 'LEAF_1', 'amount': rnd(2400, 300), 'delay_minutes': random.randint(15,25)},
                    {'from': 'MID_L', 'to': 'LEAF_2', 'amount': rnd(2400, 300), 'delay_minutes': random.randint(18,28)},
                    {'from': 'MID_R', 'to': 'LEAF_3', 'amount': rnd(2400, 300), 'delay_minutes': random.randint(20,30)},
                    {'from': 'MID_R', 'to': 'LEAF_4', 'amount': rnd(2400, 300), 'delay_minutes': random.randint(22,32)},
                ]
            },
            7: {
                'strategy': f'Time-delayed micro transfers',
                'transactions': [
                    {'from': 'SRC', 'to': f'ACC_{chr(65+i)}', 'amount': random.randint(800, 1400), 'delay_minutes': i*random.randint(15,30)}
                    for i in range(random.randint(5, 8))
                ]
            },
            8: {
                'strategy': 'U-turn layering',
                'transactions': [
                    {'from': 'ORG',   'to': 'RELAY_1', 'amount': rnd(9500, 300), 'delay_minutes': 0},
                    {'from': 'RELAY_1','to': 'RELAY_2', 'amount': rnd(9000, 300), 'delay_minutes': random.randint(8,15)},
                    {'from': 'RELAY_2','to': 'RELAY_3', 'amount': rnd(8500, 300), 'delay_minutes': random.randint(20,30)},
                    {'from': 'RELAY_3','to': 'RELAY_2', 'amount': rnd(4000, 200), 'delay_minutes': random.randint(35,45)},
                    {'from': 'RELAY_2','to': 'FINAL',   'amount': rnd(7800, 400), 'delay_minutes': random.randint(50,65)},
                ]
            },
        }

        fb = patterns[fb_key]
        return {
            'attack_id':    f'ATK_{self.attack_count}_FALLBACK',
            'strategy':     fb['strategy'],
            'transactions': fb['transactions'],
            'difficulty':   difficulty,
            'is_fallback':  True,
        }

    def get_memory_state(self):
        return {
            'successful_evasions': self.successful_evasions,
            'failed_attacks':      [f['strategy'] for f in self.failed_attacks],
            'known_rules':         self.known_rules,
            'attack_count':        self.attack_count,
        }