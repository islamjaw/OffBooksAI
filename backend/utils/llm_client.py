import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json
import re

load_dotenv()

class LLMClient:
    def __init__(self):
        # Swap to AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY', 'test'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        self.model = os.getenv('MODEL_NAME', 'openai/gpt-oss-120b')
    
    # Make generate async
    async def generate(self, prompt, system_prompt='You are a helpful assistant.', max_tokens=500, temperature=0.7):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f'Error: {str(e)}'
    
    # Make generate_json async
    async def generate_json(self, prompt, system_prompt='You are a helpful assistant.', max_tokens=500):
        json_system = system_prompt + '\n\nRespond ONLY with valid JSON. No markdown, no extra text.'
        response = await self.generate(prompt, json_system, max_tokens, temperature=0.5)
        
        text = response.strip()
        if '`json' in text:
            match = re.search(r'`json\s*(\{.*?\})\s*`', text, re.DOTALL)
            if match:
                text = match.group(1)
        elif '`' in text:
            match = re.search(r'`\s*(\{.*?\})\s*`', text, re.DOTALL)
            if match:
                text = match.group(1)
        
        start = text.find('{')
        if start != -1:
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{': 
                    brace_count += 1
                elif text[i] == '}': 
                    brace_count -= 1
                    if brace_count == 0:
                        text = text[start:i+1]
                        break
        try:
            return json.loads(text)
        except:
            return {'error': 'Parse failed', 'raw': response}
    
    # Make stream async
    async def stream(self, prompt, system_prompt='You are a helpful assistant.', max_tokens=500):
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True
            )
            # Use async for
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f'Error: {str(e)}'