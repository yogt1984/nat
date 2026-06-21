"""`nat macro` — daily macro signals (SMA crosses, MACD, RSI)."""

from __future__ import annotations

import sys

from cli.common import ROOT, BOLD, W, G, R, _p


def cmd_macro(args):
    """Show macro signals for all symbols."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from data.macro import fetch_candles, add_indicators

    symbols = ["BTC", "ETH", "SOL"]
    print(f"\n  {BOLD}Macro Signals (Daily){W}\n")
    for sym in symbols:
        try:
            df = fetch_candles(sym, interval="1d", days=365)
            df = add_indicators(df)
            row = df.iloc[-1]
            cross_7_21 = "UP" if row.get("cross_7_21", 0) > 0 else "DN"
            cross_50_200 = "UP" if row.get("cross_50_200", 0) > 0 else "DN"
            rsi = row.get("rsi_14", 0)
            macd = "BUY" if row.get("macd_cross", 0) > 0 else "SELL"
            price = row.get("close", 0)
            c7 = G if cross_7_21 == "UP" else R
            c50 = G if cross_50_200 == "UP" else R
            print(f"  {BOLD}{sym}{W}  ${price:,.2f}")
            print(f"    SMA 7/21:  {c7}{cross_7_21}{W}   SMA 50/200: {c50}{cross_50_200}{W}   MACD: {macd}   RSI: {rsi:.0f}")
            print(f"    SMA7: {row.sma_7:,.2f}  SMA21: {row.sma_21:,.2f}  SMA50: {row.sma_50:,.2f}  SMA200: {row.sma_200:,.2f}")
            if "support_20" in df.columns:
                print(f"    Support: ${row.support_20:,.2f}  Resist: ${row.resist_20:,.2f}")
            print()
        except Exception as e:
            _p("x", R, f"{sym}: {e}")
    print(f"  {BOLD}Legend{W}: 7/21 = short-term trend, 50/200 = macro trend (golden/death cross)\n")


def register(sub):
    sub.add_parser('macro', help='Daily macro signals').set_defaults(func=cmd_macro)


__all__ = ["cmd_macro", "register"]
