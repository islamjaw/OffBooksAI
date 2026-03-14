import os
import json
import re
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI 

load_dotenv()

# ── Try IBM watsonx.ai first, fall back to OpenAI if not configured ──────
WATSONX_API_KEY    = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL        = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

USE_WATSONX = bool(WATSONX_API_KEY and WATSONX_PROJECT_ID)

if USE_WATSONX:
    try:
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
        print('[LLMClient] IBM watsonx.ai SDK loaded — using Granite')
    except ImportError:
        USE_WATSONX = False
        print('[LLMClient] ibm-watsonx-ai not installed — falling back to OpenAI')

if not USE_WATSONX:
    from openai import AsyncOpenAI
    print('[LLMClient] Using OpenAI-compatible endpoint (HuggingFace)')


class LLMClient:
    def __init__(self):
        if USE_WATSONX:
            self._init_watsonx()
        else:
            self._init_openai()

    def _init_watsonx(self):
        global USE_WATSONX
        try:
            credentials = Credentials(
                api_key=WATSONX_API_KEY,
                url=WATSONX_URL,
            )
            self._wx_client   = APIClient(credentials=credentials)
            self._wx_model_id = os.getenv('WATSONX_MODEL_ID', 'ibm/granite-13b-chat-v2')
            self._wx_project  = WATSONX_PROJECT_ID
            print(f'[LLMClient] watsonx.ai ready — model: {self._wx_model_id}')
        except Exception as e:
            print(f'[LLMClient] watsonx.ai failed ({e.__class__.__name__}) — falling back to OpenAI')
            USE_WATSONX = False
            self._init_openai()

    def _init_openai(self):
        """Fallback: OpenAI-compatible endpoint (HuggingFace GPT-OSS-120B)."""
        self._oai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY', 'test'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        self._oai_model = os.getenv('MODEL_NAME', 'openai/gpt-oss-120b')

    # ── Primary generate call ─────────────────────────────────────────
    async def generate(self, prompt: str, system_prompt: str = 'You are a helpful assistant.',
                       max_tokens: int = 500, temperature: float = 0.7) -> str:
        if USE_WATSONX:
            return await self._wx_generate(prompt, system_prompt, max_tokens, temperature)
        return await self._oai_generate(prompt, system_prompt, max_tokens, temperature)

    # ── JSON generate call ────────────────────────────────────────────
    async def generate_json(self, prompt: str,
                            system_prompt: str = 'You are a helpful assistant.',
                            max_tokens: int = 500) -> dict:
        json_system = system_prompt + '\n\nRespond ONLY with valid JSON. No markdown, no extra text.'
        response    = await self.generate(prompt, json_system, max_tokens, temperature=0.3)
        return self._parse_json(response)

    # ── Streaming call ────────────────────────────────────────────────
    async def stream(self, prompt: str, system_prompt: str = 'You are a helpful assistant.',
                     max_tokens: int = 500):
        """Yields text chunks. Falls back to chunking a full response if streaming unavailable."""
        if USE_WATSONX:
            # watsonx.ai ModelInference doesn't support true async streaming yet —
            # generate the full response and simulate streaming chunk by chunk
            full = await self._wx_generate(prompt, system_prompt, max_tokens, temperature=0.7)
            chunk_size = 4
            for i in range(0, len(full), chunk_size):
                yield full[i:i+chunk_size]
                await asyncio.sleep(0.01)
        else:
            async for chunk in self._oai_stream(prompt, system_prompt, max_tokens):
                yield chunk

    # ── watsonx.ai internals ──────────────────────────────────────────
    async def _wx_generate(self, prompt: str, system_prompt: str,
                            max_tokens: int, temperature: float) -> str:
        """Run watsonx.ai inference in a thread so it doesn't block the event loop."""
        def _sync_call():
            model = ModelInference(
                model_id=self._wx_model_id,
                api_client=self._wx_client,
                project_id=self._wx_project,
                params={
                    Params.MAX_NEW_TOKENS:  max_tokens,
                    Params.TEMPERATURE:     temperature,
                    Params.REPETITION_PENALTY: 1.1,
                }
            )
            # Granite chat format: system + user turn
            full_prompt = (
                f'<|system|>\n{system_prompt}\n'
                f'<|user|>\n{prompt}\n'
                f'<|assistant|>\n'
            )
            result = model.generate_text(prompt=full_prompt)
            return result if isinstance(result, str) else str(result)

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, _sync_call)
            return text.strip()
        except Exception as e:
            print(f'[LLMClient] watsonx.ai error: {e}')
            return f'Error: {str(e)}'

    # ── OpenAI-compatible internals ───────────────────────────────────
    async def _oai_generate(self, prompt: str, system_prompt: str,
                             max_tokens: int, temperature: float) -> str:
        try:
            response = await self._oai_client.chat.completions.create(
                model=self._oai_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f'Error: {str(e)}'

    async def _oai_stream(self, prompt: str, system_prompt: str, max_tokens: int):
        try:
            stream = await self._oai_client.chat.completions.create(
                model=self._oai_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f'Error: {str(e)}'

    # ── JSON parser ───────────────────────────────────────────────────
    def _parse_json(self, response: str) -> dict:
        text = response.strip()

        # Strip markdown fences
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)

        # Extract outermost { ... }
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
        except Exception:
            return {'error': 'Parse failed', 'raw': response}