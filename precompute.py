import json
import os
import time
from main import score_company

sector_tickers = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "AMD", "INTC",
        "QCOM", "TXN", "IBM", "NOW", "ADBE", "CRM", "INTU", "AMAT",
        "MU", "LRCX", "KLAC", "SNPS", "CDNS", "ADI", "MRVL", "FTNT",
        "PANW", "CRWD", "DDOG", "WDAY", "DELL", "HPE", "NTAP", "ANSS",
        "PTC", "KEYS", "ON", "MPWR", "GLW", "APH", "TEL", "CTSH",
        "ACN", "AKAM", "VRSN", "PLTR", "ANET"
    ],

    "Healthcare": [
        "JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
        "BMY", "AMGN", "PFE", "GILD", "VRTX", "REGN", "CI", "CVS",
        "HUM", "SYK", "BSX", "MDT", "ISRG", "ELV", "ZTS", "HCA",
        "MCK", "CAH", "IDXX", "IQV", "RMD", "DXCM", "ALGN", "BAX",
        "BDX", "EW", "WAT", "MTD", "WST", "HOLX", "ZBH", "PODD",
        "CNC", "MOH", "BIIB", "DGX", "LH"
    ],

    "Financial Services": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",
        "BLK", "SCHW", "USB", "PNC", "COF", "TFC", "MTB", "FITB",
        "KEY", "HBAN", "RF", "CFG", "STT", "NTRS", "BK", "CME",
        "ICE", "NDAQ", "MCO", "SPGI", "MSCI", "CBOE", "AMP", "MMC",
        "AON", "AJG", "PGR", "TRV", "ALL", "CB", "AIG", "MET",
        "PRU", "AFL", "HIG", "WRB", "CINF", "MA", "V", "PYPL"
    ],

    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "BKNG",
        "LOW", "ABNB", "MAR", "YUM", "DHI", "F", "GM", "EBAY",
        "RCL", "CCL", "LVS", "NCLH", "WYNN", "MGM", "HLT", "CMG",
        "DPZ", "DRI", "QSR", "TXRH", "ROST", "BURL", "DKS", "RL",
        "TPR", "DECK", "CROX", "LULU", "ULTA", "WSM", "LEN", "PHM",
        "NVR", "TOL", "KBH", "APTV", "BWA"
    ],

    "Industrials": [
        "GE", "CAT", "RTX", "HON", "UPS", "LMT", "DE", "BA",
        "MMM", "GD", "NOC", "EMR", "ETN", "ITW", "PH", "CMI",
        "ROK", "XYL", "AME", "FTV", "DOV", "IEX", "IR", "WAB",
        "WM", "RSG", "CSX", "UNP", "NSC", "JBHT", "ODFL", "XPO",
        "FDX", "DAL", "UAL", "LUV", "HII", "LDOS", "LHX", "TDG",
        "HEI", "AXON", "PWR", "JCI", "CARR"
    ],

    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
        "OXY", "KMI", "WMB", "HAL", "DVN", "BKR", "FANG", "APA",
        "EQT", "CTRA", "TRGP", "OKE", "ET", "EPD", "MPLX", "WES",
        "ENB", "TRP", "SU", "CNQ", "IMO", "CVE", "AR", "CNX",
        "SM", "MTDR", "RRC", "CIVI", "CHRD", "MUR", "NOG", "TALO",
        "KOS", "EQNR", "SHEL", "BP", "TTE"
    ],

    "Consumer Defensive": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ",
        "CL", "KHC", "GIS", "K", "HSY", "CAG", "CPB", "SJM",
        "HRL", "MKC", "CHD", "CLX", "KMB", "KDP", "STZ", "TAP",
        "MNST", "KVUE", "KR", "SFM", "TGT", "DG", "DLTR", "BJ",
        "CASY", "SYY", "ADM", "BG", "INGR", "LW", "TSN", "PPC"
    ],

    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "CHTR", "PARA", "WBD", "OMC", "IPG", "FOXA", "FOX", "NWSA",
        "NWS", "LYV", "EA", "TTWO", "MTCH", "PINS", "SNAP", "RBLX",
        "ROKU", "SPOT", "ZM", "SIRI", "IHRT", "CARS", "ANGI", "YELP",
        "TRIP", "EXPE"
    ],

    "Real Estate": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "O", "PSA",
        "EXR", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "NNN",
        "KIM", "REG", "FRT", "BXP", "ESS", "HST", "IRM", "CBRE",
        "CCI", "SBAC", "WY", "INVH", "AMH", "SUI", "ELS", "CPT",
        "OHI", "VNO", "SLG"
    ],

    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG",
        "XEL", "ED", "ETR", "FE", "PPL", "EIX", "CMS", "DTE",
        "AES", "NI", "CNP", "WEC", "ATO", "LNT", "EVRG", "PNW",
        "NWE", "AVA", "IDA", "BKH", "OGE", "SR", "NJR", "SWX",
        "UGI", "OGS", "MGEE"
    ],

    "Basic Materials": [
        "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "DOW", "DD",
        "PPG", "ALB", "MOS", "CF", "IFF", "CE", "EMN", "FMC",
        "RPM", "SON", "PKG", "IP", "AVY", "BALL", "CCK", "AA",
        "X", "CLF", "STLD", "RS", "CMC", "MP", "SCCO", "TECK",
        "VALE", "RIO", "BHP", "GOLD", "AEM", "KGC", "AU", "CDE"
    ],
}

def precompute():
    if not os.path.exists("fraud.pt"):
        print("fraud.pt not found. Train the model first.")
        return

    peer_scores = {}

    for sector, tickers in sector_tickers.items():
        print(f"\n{sector} ({len(tickers)} tickers)")
        dist = {"fraud": [], "distress": [], "credit": [], "manip": []}

        for ticker in tickers:
            try:
                result = score_company(ticker)
            except Exception as e:
                continue

            if "error" in result:
                print(f"  skip {ticker}: {result['error']}")
                continue

            scores = result["scores"]
            for k in dist:
                dist[k].append(scores[k])
            print(f"  {ticker}  fraud={scores['fraud']:.4f}  distress={scores['distress']:.4f}  credit={scores['credit']:.4f}  manip={scores['manip']:.4f}")
            time.sleep(0.5)

        peer_scores[sector] = dist
        print(f"  {len(dist['fraud'])} companies scored")

    with open(os.path.join(os.path.dirname(__file__), "peer_scores.json"), "w") as f:
        json.dump(peer_scores, f)

    print("\nsaved peer_scores.json")
    for sector, d in peer_scores.items():
        print(f"  {sector}: {len(d['fraud'])} companies")


if __name__ == "__main__":
    precompute()
