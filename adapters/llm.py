import json
from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    @abstractmethod
    def score_sentiment(self, text: str) -> dict:
        """Returns {"sentiment": -1|0|1, "confidence": float, "reason": str}."""
        ...

    @abstractmethod
    def narrate_signal(self, context: dict) -> str:
        """Returns 3-4 sentence plain English explanation of a trading signal."""
        ...


class OpenAIAdapter(LLMAdapter):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def score_sentiment(self, text: str) -> dict:
        prompt = (
            "Analyze the sentiment of this crypto-related text toward Bitcoin/Ethereum price "
            "direction.\n\n"
            f"Text: {text}\n\n"
            "Respond with ONLY valid JSON in this exact format:\n"
            '{"sentiment": -1, "confidence": 0.8, "reason": "one sentence explanation"}\n\n'
            "sentiment must be exactly -1 (bearish), 0 (neutral), or 1 (bullish).\n"
            "confidence must be a float between 0.0 and 1.0."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.choices[0].message.content.strip())

    def narrate_signal(self, context: dict) -> str:
        features = "\n".join(f"- {f}" for f in context["top_features"])
        prices = ", ".join(str(p) for p in context["recent_prices"])
        prompt = (
            "You are a crypto trading assistant. Explain this trading signal in 3-4 plain "
            "English sentences.\n\n"
            f"Signal: {context['signal']} for {context['asset']}\n"
            f"Confidence: {context['confidence']:.0%}\n"
            f"Top factors:\n{features}\n"
            f"Recent sentiment: {context['sentiment_summary']}\n"
            f"Last 3 prices (4h closes, USD): {prices}\n\n"
            "Be specific and factual. Do not give financial advice."
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
