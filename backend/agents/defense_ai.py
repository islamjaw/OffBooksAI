import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient


class DefenseAI(BaseAgent):
    def __init__(self, ring_scout):
        super().__init__('DefenseAI')
        self.llm = LLMClient()
        self.ring_scout = ring_scout       # reference so we can add rules to it
        self.adaptations = []              # history of every rule change made
        self.evasion_count = 0

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    async def execute(self, input_data):
        """
        Called when FraudGPT successfully evades Ring Scout.

        input_data = {
            'attack': { ...FraudGPT attack dict... },
            'evasion_reason': 'circular routing not detected'  # optional
        }

        Returns a dict describing the new rule that was added.
        """
        self.evasion_count += 1
        attack = input_data.get('attack', {})
        evasion_reason = input_data.get('evasion_reason', 'unknown evasion method')

        self.log(f'Evasion #{self.evasion_count} detected! '
                 f'Attack: {attack.get("strategy", "?")} | '
                 f'Reason: {evasion_reason}')

        adaptation = await self._generate_new_rule(attack, evasion_reason)

        # Apply the new rule to Ring Scout immediately
        new_rule_name = adaptation.get('rule_name', '')
        if new_rule_name:
            self.ring_scout.add_rule(new_rule_name)

        self.adaptations.append(adaptation)
        return adaptation

    # ------------------------------------------------------------------
    # LLM-powered rule generation
    # ------------------------------------------------------------------
    async def _generate_new_rule(self, attack, evasion_reason):
        strategy = attack.get('strategy', 'unknown')
        transactions = attack.get('transactions', [])

        txn_summary = '\n'.join(
            f"  {t.get('from')} -> {t.get('to')}: "
            f"${t.get('amount', 0)} (delay: {t.get('delay_minutes', 0)} min)"
            for t in transactions[:8]   # cap at 8 to keep prompt short
        )

        prompt = f"""A fraud attack EVADED our detection system.

Attack strategy: {strategy}
Why it evaded: {evasion_reason}

Transactions used:
{txn_summary}

Propose ONE new graph-based detection rule to catch this pattern in future.
Output ONLY this JSON:
{{
    "rule_name": "short_snake_case_name",
    "description": "one sentence explaining what the rule detects",
    "detection_logic": "describe the graph property to check (e.g. look for cycles > 3 hops)",
    "confidence": 85
}}"""

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt=(
                'You are a fraud detection engineer. '
                'Analyze evasion patterns and propose precise graph-based rules. '
                'Output only valid JSON.'
            ),
            max_tokens=300
        )

        if 'error' in result:
            return self._fallback_rule(strategy)

        return {
            'rule_name': result.get('rule_name', f'rule_{self.evasion_count}'),
            'description': result.get('description', 'Auto-generated rule'),
            'detection_logic': result.get('detection_logic', ''),
            'confidence': result.get('confidence', 70),
            'triggered_by': strategy,
            'evasion_number': self.evasion_count
        }

    # ------------------------------------------------------------------
    # Fallback if LLM fails
    # ------------------------------------------------------------------
    def _fallback_rule(self, strategy):
        fallback_rules = {
            'fan_out': {
                'rule_name': 'multi_hop_fan_out',
                'description': 'Detect fan-out patterns across 2+ hops',
                'detection_logic': 'Check if any second-degree node receives from 3+ sources',
            },
            'circular': {
                'rule_name': 'long_cycle_detection',
                'description': 'Detect circular routing with 4+ intermediaries',
                'detection_logic': 'Find simple cycles with path length > 3',
            },
            'default': {
                'rule_name': f'adaptive_rule_{self.evasion_count}',
                'description': 'Adaptive rule generated from evasion event',
                'detection_logic': 'Flag clusters with unusual edge density',
            }
        }

        key = 'circular' if 'circular' in strategy.lower() else \
              'fan_out' if 'fan' in strategy.lower() else 'default'

        rule = fallback_rules[key].copy()
        rule['confidence'] = 65
        rule['triggered_by'] = strategy
        rule['evasion_number'] = self.evasion_count
        return rule