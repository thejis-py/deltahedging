"""Data access: pull market data from the Settrade Open API, cache it under
data/raw/, and load it back as pandas DataFrames.

Credentials come from environment variables (or a gitignored .env file in the
project root). Nothing secret is stored in this file.

NOT YET TESTED against the live API. See the assumptions list in the README/notes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TZ = "Asia/Bangkok"

load_dotenv(ROOT / ".env")  # silently does nothing if the file doesn't exist
log = logging.getLogger(__name__)

REQUIRED_ENV = (
    "SETTRADE_APP_ID",
    "SETTRADE_APP_SECRET",
    "SETTRADE_BROKER_ID",
    "SETTRADE_APP_CODE",
)
OHLCV = ["open", "high", "low", "close", "volume", "value"]


# ------------------------------------------------------------------ client
def _credentials() -> dict:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}.\n"
            "Copy .env.example to .env and fill in your values, "
            "or export them in your shell."
        )
    return {k: os.environ[k] for k in REQUIRED_ENV}


def get_market_client():
    """Authenticated MarketData client. The SDK is imported lazily so that
    loading cached files works even where settrade-v2 is not installed."""
    from settrade_v2 import Investor

    c = _credentials()
    investor = Investor(
        app_id=c["SETTRADE_APP_ID"],
        app_secret=c["SETTRADE_APP_SECRET"],
        broker_id=c["SETTRADE_BROKER_ID"],
        app_code=c["SETTRADE_APP_CODE"],
        is_auto_queue=False,
    )
    return investor.MarketData()


# ------------------------------------------------------------------ helpers
def _call_with_retry(fn, *, retries: int = 3, base_delay: float = 1.0, **kwargs):
    """Retry transient failures (network, rate limit) with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            return fn(**kwargs)
        except Exception as exc:  # SDK exception types not known; keep broad
            if attempt == retries:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("Call failed (%s). Retry %d/%d in %.1fs",
                        exc, attempt, retries - 1, delay)
            time.sleep(delay)


def _to_datetime(values) -> pd.DatetimeIndex:
    """Convert API timestamps to tz-aware Asia/Bangkok.
    Handles epoch seconds, epoch milliseconds, or ISO strings."""
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.abs().max() > 1e11 else "s"
        idx = pd.to_datetime(s, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(s, utc=True)
    return pd.DatetimeIndex(idx).tz_convert(TZ)


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text))


def _paths(symbol: str, interval: str, start: str, end: str | None):
    stem = f"{_safe(symbol)}_{_safe(interval)}_{_safe(start[:10])}_{_safe(end[:10]) if end else 'latest'}"
    return RAW_DIR / f"{stem}.csv", RAW_DIR / f"{stem}.meta.json"


# ------------------------------------------------------------------ fetch
def fetch_candles(symbol: str, start: str, end: str | None = None,
                  interval: str = "1d", limit: int = 1000,
                  market=None) -> pd.DataFrame:
    """Download OHLCV candles for one symbol (stock, index, DW, ...).

    start/end: ISO strings like "2023-01-01T00:00".
    interval : e.g. "1d", "1h"-style codes accepted by the API (check docs).
    Returns a DataFrame indexed by Asia/Bangkok timestamps.
    """
    market = market or get_market_client()
    kwargs = dict(symbol=symbol, interval=interval, limit=limit,
                  start=start, normalized=True)
    if end:
        kwargs["end"] = end

    raw = _call_with_retry(market.get_candlestick, **kwargs)

    # Expected: dict of equal-length lists {"time": [...], "open": [...], ...}
    cols = {k: v for k, v in raw.items() if isinstance(v, (list, tuple))}
    if "time" not in cols or "close" not in cols:
        raise ValueError(f"Unexpected candlestick response keys: {list(raw)}")

    df = pd.DataFrame(cols)
    df.index = _to_datetime(df.pop("time"))
    df.index.name = "datetime"
    df = df[[c for c in OHLCV if c in df.columns]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if len(df) >= limit:
        log.warning("%s: received %d rows == limit; the history may be truncated. "
                    "Narrow the date range or raise `limit`.", symbol, len(df))
    if df.empty:
        log.warning("%s: no data returned for %s -> %s", symbol, start, end)
    return df


def get_quote(symbol: str, market=None) -> dict:
    """Latest quote snapshot (raw dict from the API)."""
    market = market or get_market_client()
    return _call_with_retry(market.get_quote_symbol, symbol=symbol)


# ------------------------------------------------------------- cache / load
def save_raw(df: pd.DataFrame, symbol: str, interval: str,
             start: str, end: str | None) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _paths(symbol, interval, start, end)
    df.to_csv(csv_path)
    meta = {
        "symbol": symbol, "interval": interval, "start": start, "end": end,
        "rows": len(df),
        "first": str(df.index.min()) if len(df) else None,
        "last": str(df.index.max()) if len(df) else None,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Settrade Open API (settrade-v2)",
        "broker_id": os.environ.get("SETTRADE_BROKER_ID"),  # not a secret; shows sandbox vs live
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return csv_path


def load_raw(symbol: str, start: str, end: str | None = None,
             interval: str = "1d") -> pd.DataFrame:
    """Load a previously downloaded file (no API call, no credentials needed)."""
    csv_path, _ = _paths(symbol, interval, start, end)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run get_prices(...) first.")
    df = pd.read_csv(csv_path, index_col="datetime")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    return df


def get_prices(symbol: str, start: str, end: str | None = None,
               interval: str = "1d", refresh: bool = False,
               limit: int = 1000, market=None) -> pd.DataFrame:
    """Main entry point: use the cached file if present, otherwise download and cache.

    With end=None the cache holds "up to the time of download"; pass refresh=True
    to update it. Use an explicit `end` for reproducible backtests.
    """
    csv_path, _ = _paths(symbol, interval, start, end)
    if csv_path.exists() and not refresh:
        return load_raw(symbol, start, end, interval)
    df = fetch_candles(symbol, start, end, interval, limit, market)
    if not df.empty:
        save_raw(df, symbol, interval, start, end)
    return df


def get_many(symbols: list[str], start: str, end: str | None = None,
             interval: str = "1d", refresh: bool = False,
             pause: float = 0.3) -> dict[str, pd.DataFrame]:
    """Download several symbols (e.g. a DW and its underlying), reusing one client.
    A failure on one symbol is logged and skipped, not fatal."""
    market = None
    out = {}
    for sym in symbols:
        csv_path, _ = _paths(sym, interval, start, end)
        try:
            if not (csv_path.exists() and not refresh) and market is None:
                market = get_market_client()
            out[sym] = get_prices(sym, start, end, interval, refresh, market=market)
        except Exception as exc:
            log.error("Failed to get %s: %s", sym, exc)
        time.sleep(pause)  # be gentle with rate limits
    return out


# --------------------------------------------------------------------- CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download Settrade candles to data/raw/")
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--start", required=True, help='e.g. "2023-01-01T00:00"')
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    for sym, d in get_many(a.symbols, a.start, a.end, a.interval, a.refresh).items():
        print(f"{sym}: {len(d)} rows"
              + (f", {d.index.min():%Y-%m-%d} -> {d.index.max():%Y-%m-%d}" if len(d) else ""))

"""
USAGE:# from the project root, venv active
python -m src.data AOT --start 2023-01-01T00:00 --end 2024-12-31T00:00

python
from src.data import get_prices, get_many

spot = get_prices("AOT", "2023-01-01T00:00", "2024-12-31T00:00")
dw   = get_prices("AOT19C2512A", "2024-06-01T00:00", "2024-12-31T00:00")  # example DW symbol
returns = spot["close"].pct_change().dropna()
"""
