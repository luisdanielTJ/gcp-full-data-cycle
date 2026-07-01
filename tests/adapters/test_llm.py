from unittest.mock import MagicMock, patch

import pytest

from adapters.llm import OpenAIAdapter


@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        adapter = OpenAIAdapter(api_key="fake-key")
        yield adapter


def _openai_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


def test_openai_score_sentiment_bullish(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        '{"sentiment": 1, "confidence": 0.85, "reason": "Positive ETF approval news"}'
    )
    result = mock_openai.score_sentiment("Bitcoin ETF approved by the SEC today")
    assert result["sentiment"] == 1
    assert result["confidence"] == pytest.approx(0.85)
    assert isinstance(result["reason"], str)


def test_openai_score_sentiment_bearish(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        '{"sentiment": -1, "confidence": 0.72, "reason": "Regulatory crackdown fears"}'
    )
    result = mock_openai.score_sentiment("SEC sues major crypto exchange for fraud")
    assert result["sentiment"] == -1
    assert 0.0 <= result["confidence"] <= 1.0


def test_openai_score_sentiment_validates_range(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        '{"sentiment": 0, "confidence": 0.5, "reason": "No strong signal"}'
    )
    result = mock_openai.score_sentiment("Crypto markets quiet today")
    assert result["sentiment"] in (-1, 0, 1)
    assert 0.0 <= result["confidence"] <= 1.0


def test_openai_narrate_signal_returns_non_empty_string(mock_openai):
    mock_openai.client.chat.completions.create.return_value = _openai_response(
        "BTC shows a BUY signal at 71% confidence driven by MACD crossover and positive sentiment."
    )
    context = {
        "signal": "BUY",
        "asset": "BTC",
        "confidence": 0.71,
        "top_features": ["RSI(14)=42: not overbought", "MACD: bullish crossover 8h ago"],
        "sentiment_summary": "Positive 24h score: 0.6, ETF inflow mentions up",
        "recent_prices": [48000.0, 49500.0, 50200.0],
    }
    result = mock_openai.narrate_signal(context)
    assert isinstance(result, str)
    assert len(result) > 20
