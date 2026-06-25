import json
import time
import numpy as np
import yfinance as yf
from main import score_company, peer_scores_path

sector_tickers = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO",
        "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "ADBE", "CRM", "INTU",
        "AMAT", "MU", "LRCX"
    ],
    "Healthcare": [
        "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "PFE", "GILD", "VRTX", "REGN", "CI", "CVS", "HUM", "SYK",
        "BSX", "MDT"
    ],
    "Financial Services": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
        "SCHW", "USB", "PNC", "COF", "TFC", "MTB", "FITB", "KEY", "HBAN",
        "RF", "CFG"
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "BKNG", "LOW",
        "ABNB", "MAR", "YUM", "DHI", "F", "GM", "EBAY", "ETSY", "RCL",
        "CCL", "LVS"
    ],
    "Industrials": [
        "GE", "CAT", "RTX", "HON", "UPS", "LMT", "DE", "BA", "MMM",
        "GD", "NOC", "EMR", "ETN", "ITW", "PH", "CMI", "ROK", "XYL",
        "AME", "FTV"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "PXD",
        "OXY", "KMI", "WMB", "HAL", "DVN", "HES", "BKR", "FANG", "MRO",
        "APA", "EQT"
    ],
    "Consumer Defensive": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "CL",
        "KHC", "GIS", "K", "HSY", "CAG", "CPB", "SJM", "HRL", "MKC",
        "CHD", "CLX"
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "CHTR", "PARA", "WBD", "OMC", "IPG", "FOXA", "NWSA", "LYV",
        "EA", "TTWO", "MTCH"
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "O", "PSA", "EXR",
        "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "NNN", "KIM", "REG",
        "FRT", "BXP"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "XEL",
        "ED", "ETR", "FE", "PPL", "EIX", "CMS", "DTE", "AES", "NI",
        "CNP", "WEC"
    ],
    "Basic Materials": [
        "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "DOW", "DD", "PPG",
        "ALB", "MOS", "CF", "IFF", "CE", "EMN", "FMC", "RPM", "SON",
        "SEE", "PKG"
    ],
}


def precompute():
    import os
    if not os.path.exists("fraud.pt"):
        print("fraud.pt not found. Train the model first.")
        return

    scores = {}

    for sector, tickers in sector_tickers.items():
        print(f"\n{sector} ({len(tickers)} tickers)")
        dist = {"fraud": [], "distress": [], "credit": [], "manip": []}

        for ticker in tickers:
            try:
                result = score_company(ticker)
                if "error" in result:
                    print(f"  skip {ticker}: {result['error']}")
                    continue
                raw = result["raw"]
                for k in dist:
                    dist[k].append(raw[k])
                print(f"  {ticker}  fraud={raw['fraud']:.4f}  distress={raw['distress']:.4f}  credit={raw['credit']:.4f}  manip={raw['manip']:.4f}")
                time.sleep(0.5)
            except Exception as e:
                print(f"  error {ticker}: {e}")

        scores[sector] = dist
        print(f"  {len(dist['fraud'])} companies scored")

    with open(peer_scores_path, "w") as f:
        json.dump(scores, f)

    print(f"\nsaved peer_scores.json")
    for sector, d in scores.items():
        print(f"  {sector}: {len(d['fraud'])} companies")


if __name__ == "__main__":
    precompute()