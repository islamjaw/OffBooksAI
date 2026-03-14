import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient

class InvestigationAgent(BaseAgent):
    def __init__(self):
        super().__init__('Investigation Agent')
        self.llm = LLMClient()
    
    async def execute(self, ring_data):
        self.log(f'Investigating {ring_data.get("ring_id")}')
        
        accounts = ', '.join(ring_data.get('accounts', []))
        patterns = ', '.join(ring_data.get('patterns', []))
        
        prompt = f'''Analyze this fraud ring:

Ring ID: {ring_data.get('ring_id')}
Accounts: {accounts}
Suspicion Score: {ring_data.get('suspicion_score', 0)}/100
Patterns: {patterns}
Total Amount: ${ring_data.get('total_amount', 0):,.2f}
Timeframe: {ring_data.get('timeframe_hours', 0)} hours

Write a brief fraud investigation report with:
1. Executive summary (2 sentences)
2. Key evidence
3. Recommended actions'''

        report = await self.llm.generate(
            prompt=prompt,
            system_prompt='You are a financial crimes investigator. Be professional and concise.',
            max_tokens=400
        )
        
        return {
            'ring_id': ring_data['ring_id'],
            'report': report,
            'timestamp': ring_data.get('timestamp', '')
        }
    
    async def stream_report(self, ring_data):
        accounts = ', '.join(ring_data.get('accounts', []))
        
        prompt = f'''Analyze this fraud ring:

Ring ID: {ring_data.get('ring_id')}
Accounts: {accounts}
Suspicion Score: {ring_data.get('suspicion_score', 0)}/100

Write a 3-sentence fraud investigation report.'''

        async for chunk in self.llm.stream(
            prompt=prompt,
            system_prompt='You are a fraud investigator. Be concise.',
            max_tokens=200
        ):
            yield chunk
