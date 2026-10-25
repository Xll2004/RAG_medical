# llm_client.py
# Unified LLM call: --provider claude uses Claude Sonnet 4.6, --provider gpt uses GPT-4o-mini
# Both at temperature=0, with retries
import os, time
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-4o-mini"


class LLM:
    def __init__(self, provider):
        self.provider = provider
        if provider == "claude":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = CLAUDE_MODEL
        elif provider == "gpt":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60)
            self.model = GPT_MODEL
        else:
            raise ValueError(provider)

    def chat(self, messages, system=None, max_tokens=200):
        """messages: [{'role': 'user'|'assistant', 'content': str}, ...]"""
        for attempt in range(6):
            try:
                if self.provider == "claude":
                    # extra_body: works with both old and new anthropic SDKs (the new SDK's signature has no temperature)
                    kw = dict(model=self.model, messages=messages, max_tokens=max_tokens,
                              extra_body={"temperature": 0})
                    if system:
                        kw["system"] = system
                    r = self.client.messages.create(**kw)
                    return r.content[0].text if r.content else ""
                msgs = ([{"role": "system", "content": system}] if system else []) + messages
                r = self.client.chat.completions.create(model=self.model, messages=msgs,
                                                        max_tokens=max_tokens, temperature=0)
                return r.choices[0].message.content or ""
            except Exception as e:  # rate limit / connection / 5xx
                wait = 2 ** attempt * 3
                print(f"  API error ({type(e).__name__}: {str(e)[:80]}), retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError("API failed after retries")
