import pytest
from unittest.mock import MagicMock, patch
from adapters.llm import GeminiAdapter


@pytest.fixture
def mock_gemini():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        adapter = GeminiAdapter(api_key="fake-key")
        yield adapter


def test_score_sentiment_bullish(mock_gemini):
    mock_gemini.client.models.generate_content.return_value = MagicMock(
        text='{"sentiment": 1, "confidence": 0.85, "reason": "Positive ETF approval news"}'
    )
    result = mock_gemini.score_sentiment("Bitcoin ETF approved by the SEC today")
    assert result["sentiment"] == 1
    assert result["confidence"] == pytest.approx(0.85)
    assert isinstance(result["reason"], str)


def test_score_sentiment_bearish(mock_gemini):
    mock_gemini.client.models.generate_content.return_value = MagicMock(
        text='{"sentiment": -1, "confidence": 0.72, "reason": "Regulatory crackdown fears"}'
    )
    result = mock_gemini.score_sentiment("SEC sues major crypto exchange for fraud")
    assert result["sentiment"] == -1
    assert 0.0 <= result["confidence"] <= 1.0


def test_score_sentiment_validates_range(mock_gemini):
    mock_gemini.client.models.generate_content.return_value = MagicMock(
        text='{"sentiment": 0, "confidence": 0.5, "reason": "No strong signal"}'
    )
    result = mock_gemini.score_sentiment("Crypto markets quiet today")
    assert result["sentiment"] in (-1, 0, 1)
    assert 0.0 <= result["confidence"] <= 1.0


def test_narrate_signal_returns_non_empty_string(mock_gemini):
    mock_gemini.client.models.generate_content.return_value = MagicMock(
        text="BTC shows a BUY signal at 71% confidence driven by MACD crossover and positive sentiment."
    )
    context = {
        "signal": "BUY",
        "asset": "BTC",
        "confidence": 0.71,
        "top_features": ["RSI(14)=42: not overbought", "MACD: bullish crossover 8h ago"],
        "sentiment_summary": "Positive 24h score: 0.6, ETF inflow mentions up",
        "recent_prices": [48000.0, 49500.0, 50200.0],
    }
    result = mock_gemini.narrate_signal(context)
    assert isinstance(result, str)
    assert len(result) > 20
