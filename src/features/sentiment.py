"""Sentiment feature extraction for VRMS.

Adapted from the NSE Sentiment Analyzer.
Provides SmartScore and event flags as ML features.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


# Financial sentiment lexicon for VADER augmentation
FINANCIAL_BOOSTERS = {
    "bullish": 2.5, "bearish": -2.5, "outperform": 2.0, "underperform": -2.0,
    "overweight": 1.5, "underweight": -1.5, "upside": 1.8, "downside": -1.8,
    "buy": 1.5, "accumulate": 1.2, "reduce": -1.2, "sell": -2.0,
    "downgrade": -2.0, "upgrade": 2.0, "positive": 1.0, "negative": -1.0,
    "surge": 1.5, "plunge": -2.0, "rally": 1.5, "crash": -2.5,
    "record": 1.0, "decline": -1.0, "profit": 1.0, "loss": -1.0,
    "dividend": 1.0, "expansion": 1.0, "growth": 1.0, "slowdown": -1.5,
    "momentum": 1.0, "volatility": -0.5, "correction": -1.0, "breakout": 1.5,
    "breakdown": -1.5, "resistance": -0.3, "support": 0.3,
    "npa": -2.0, "npas": -2.0, "gnpa": -2.0, "nnpa": -1.5,
    "aum": 1.0, "pat": 1.0, "ebitda": 1.0, "nim": 1.0,
    "roe": 1.0, "roce": 1.0, "divestment": -1.0, "disinvestment": -1.0,
    "mandi": -1.5, "tezi": 1.5, "gira": -1.5, "chada": 1.5,
    "robust": 1.5, "resilient": 1.0, "stellar": 2.0, "beat": 1.5,
    "miss": -1.5, "avoid": -1.5, "headwinds": -1.5, "tailwinds": 1.5,
    "overbought": -1.0, "oversold": 1.0, "accumulation": 1.0, "distribution": -1.0,
    "oversubscribed": 1.5, "undersubscribed": -1.5, "listing": 0.5,
    "slippage": -1.5, "provisioning": -0.8, "moratorium": -1.0,
    "recapitalization": 1.2, "infusion": 1.5, "pledged": -1.0, "unpledged": 1.0,
    "inflow": 1.0, "outflow": -1.0, "buying": 1.0, "selling": -1.0,
    "doubled": 1.5, "tripled": 2.0, "multibagger": 1.5,
    "topline": 0.5, "bottomline": 1.0, "risk": 0.0, "bear": -1.5,
    "mismanagement": -2.0, "compliance": 0.5, "scrutiny": -1.0,
    "buyout": 1.0, "merger": 0.5, "acquisition": 0.5, "delisting": -0.5,
    "depreciation": -1.0, "appreciation": 1.0, "deficit": -1.0,
    "gire": -1.5, "giri": -1.5, "chade": 1.5, "chadi": 1.5,
    "tej": 1.0, "mand": -1.0,
}


def get_sentiment_analyzer():
    """Initialize VADER with custom financial lexicon."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sia.lexicon.update(FINANCIAL_BOOSTERS)
        return sia
    except Exception as e:
        logger.error(f"Failed to initialize sentiment analyzer: {e}")
        return None


def fetch_headlines(symbol: str, days: int = 5) -> list[dict]:
    """Fetch news headlines for a symbol.
    
    Args:
        symbol: NSE ticker
        days: Number of days to look back
        
    Returns:
        List of headline dicts with 'title', 'source', 'date'
    """
    headlines = []
    
    try:
        import feedparser
        
        sources = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}.NS&region=IN&lang=en-IN",
            f"https://www.moneycontrol.com/rss/results.xml",
            f"https://economictimes.indiatimes.com/markets/stocks/rssfeeds/21468427.cms",
        ]
        
        for url in sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    headlines.append({
                        'title': entry.get('title', ''),
                        'source': url.split('/')[2],
                        'date': entry.get('published', '')
                    })
            except Exception as e:
                logger.debug(f"Feed fetch failed for {url}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to fetch headlines for {symbol}: {e}")
    
    return headlines


def compute_smartscore(headline_scores: list[float]) -> float:
    """Compute SmartScore (0-100) from headline sentiment scores.
    
    Args:
        headline_scores: List of VADER compound scores
        
    Returns:
        SmartScore value (0-100)
    """
    if not headline_scores:
        return 50.0
    
    n = len(headline_scores)
    
    # Positive/negative counts
    pos_count = sum(1 for s in headline_scores if s >= 0.3)
    neg_count = sum(1 for s in headline_scores if s <= -0.3)
    
    # Average compound score
    avg_compound = sum(headline_scores) / n
    
    # Breadth: ratio of positive vs negative
    breadth = (pos_count - neg_count) / n
    
    # Volume: log-normalized headline count
    import math
    volume = min(math.log1p(n) / math.log1p(20), 1.0)
    
    # SmartScore components
    s_events = (avg_compound + 1) / 2  # Map [-1,1] to [0,1]
    s_breadth = (breadth + 1) / 2
    s_volume = volume
    
    # Composite (no recency without history)
    smartscore = 0.5 * s_events * 100 + 0.3 * s_breadth * 100 + 0.2 * s_volume * 100
    
    return max(0.0, min(100.0, smartscore))


def compute_sentiment_features(symbol: str, date: str | datetime) -> dict:
    """Compute sentiment features for a symbol on a date.
    
    Args:
        symbol: NSE ticker
        date: Date to compute features for
        
    Returns:
        Dict with sentiment features
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)
    
    # Fetch headlines
    headlines = fetch_headlines(symbol, days=5)
    
    if not headlines:
        return {
            'smartscore': 50.0,
            'sentiment_pos_count': 0,
            'sentiment_neg_count': 0,
            'sentiment_neutral_count': 0,
            'sentiment_volume': 0,
        }
    
    # Analyze sentiment
    sia = get_sentiment_analyzer()
    if sia is None:
        return {
            'smartscore': 50.0,
            'sentiment_pos_count': 0,
            'sentiment_neg_count': 0,
            'sentiment_neutral_count': 0,
            'sentiment_volume': 0,
        }
    
    scores = []
    pos_count = 0
    neg_count = 0
    neutral_count = 0
    
    for headline in headlines:
        vs = sia.polarity_scores(headline['title'])
        compound = vs['compound']
        scores.append(compound)
        
        if compound >= 0.3:
            pos_count += 1
        elif compound <= -0.3:
            neg_count += 1
        else:
            neutral_count += 1
    
    # SmartScore
    smartscore = compute_smartscore(scores)
    
    return {
        'smartscore': smartscore,
        'sentiment_pos_count': pos_count,
        'sentiment_neg_count': neg_count,
        'sentiment_neutral_count': neutral_count,
        'sentiment_volume': len(scores),
    }
