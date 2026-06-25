import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)

try:
    from entmax import sparsemax
except ImportError:
    def sparsemax(x, dim=-1):
        return torch.softmax(x, dim=dim)

base_features = [
    "act", "ap", "at", "ceq", "che",
    "cogs", "csho", "dlc", "dltis",
    "dltt", "dp", "ib", "invt",
    "ivao", "ivst", "lct", "lt",
    "ni", "pstk", "re",
    "rect", "sale", "sstk", "txp",
    "txt", "xint", "prcc_f",
    "dch_wc", "ch_rsst",
    "dch_rec", "dch_inv",
    "soft_assets", "ch_cs",
    "ch_cm", "ch_roa",
    "issue", "bm", "dpi",
    "reoa", "EBIT", "ch_fcf"
]

derived_features = [
    "current_ratio", "debt_ratio", "cash_ratio", "profit_margin",
    "ebit_margin", "asset_turnover", "retained_earnings_ratio", "working_capital_ratio"
]

temporal_features = [
    "sales_growth", "inventory_growth", "receivable_growth",
    "asset_growth", "debt_growth", "margin_change", "roa_change"
]

feature_cols = base_features + derived_features + temporal_features
seq_len = 5


class residual_block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(0.1), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class fraud_model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
            residual_block(128), nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.gru = nn.GRU(64, 128, num_layers=2, dropout=0.1, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.pool = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.shared = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1)
        )
        self.fraud_head = nn.Linear(64, 1)
        self.distress_head = nn.Linear(64, 1)
        self.credit_head = nn.Linear(64, 1)
        self.manip_head = nn.Linear(64, 1)

    def forward(self, x):
        b, t, f = x.shape
        x = self.feature_extractor(x.reshape(b * t, f)).reshape(b, t, 64)
        x, _ = self.gru(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = (x * sparsemax(self.pool(x), dim=1)).sum(dim=1)
        x = self.shared(x)
        return self.fraud_head(x), self.distress_head(x), self.credit_head(x), self.manip_head(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
scaler = RobustScaler()

model_path = os.path.join(os.path.dirname(__file__), "fraud.pt")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
peer_scores_path = os.path.join(os.path.dirname(__file__), "peer_scores.json")

if os.path.exists(model_path):
    model = fraud_model(len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

peer_scores = {}
if os.path.exists(peer_scores_path):
    with open(peer_scores_path) as f:
        peer_scores = json.load(f)


def fetch_financials(ticker):
    tk = yf.Ticker(ticker)
    bs = tk.balance_sheet
    inc = tk.income_stmt
    cf = tk.cashflow
    info = tk.info

    years = sorted(bs.columns, reverse=False)
    rows = []

    for col in years:
        def g(df, *keys):
            for k in keys:
                if k in df.index:
                    v = df.loc[k, col]
                    return float(v) if pd.notna(v) else np.nan
            return np.nan

        ni = g(inc, "Net Income")
        txt = g(inc, "Tax Provision")
        xint = g(inc, "Interest Expense")
        ebit_val = (ni or 0) + (txt or 0) + (xint or 0) if not any(v is None for v in [ni, txt, xint]) else np.nan

        rows.append({
            "fyear": col.year,
            "act": g(bs, "Current Assets"),
            "ap": g(bs, "Accounts Payable"),
            "at": g(bs, "Total Assets"),
            "ceq": g(bs, "Stockholders Equity", "Total Equity Gross Minority Interest"),
            "che": g(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
            "cogs": g(inc, "Cost Of Revenue"),
            "csho": g(bs, "Share Issued", "Common Stock Shares Outstanding"),
            "dlc": g(bs, "Current Debt", "Short Term Debt"),
            "dltis": g(cf, "Issuance Of Debt"),
            "dltt": g(bs, "Long Term Debt"),
            "dp": g(cf, "Depreciation And Amortization", "Depreciation"),
            "ib": g(inc, "Net Income Common Stockholders", "Net Income"),
            "invt": g(bs, "Inventory"),
            "ivao": g(bs, "Other Investments", "Long Term Investments"),
            "ivst": g(bs, "Short Term Investments"),
            "lct": g(bs, "Current Liabilities"),
            "lt": g(bs, "Total Liabilities Net Minority Interest", "Total Liabilities"),
            "ni": ni,
            "ppegt": g(bs, "Net PPE", "Gross PPE"),
            "pstk": g(bs, "Preferred Stock"),
            "re": g(bs, "Retained Earnings"),
            "rect": g(bs, "Receivables", "Net Receivables"),
            "sale": g(inc, "Total Revenue"),
            "sstk": g(cf, "Common Stock Issuance", "Issuance Of Capital Stock"),
            "txp": g(bs, "Taxes Payable", "Income Tax Payable"),
            "txt": txt,
            "xint": xint,
            "prcc_f": info.get("previousClose", np.nan),
            "ebit": ebit_val,
        })

    df = pd.DataFrame(rows).sort_values("fyear").reset_index(drop=True)
    return df


def compute_derived(df):
    df = df.copy()

    df["current_ratio"] = df["act"] / df["lct"]
    df["debt_ratio"] = (df["dlc"] + df["dltt"]) / df["at"]
    df["cash_ratio"] = df["che"] / df["lct"]
    df["profit_margin"] = df["ni"] / df["sale"]
    df["ebit_margin"] = df["ebit"] / df["sale"]
    df["asset_turnover"] = df["sale"] / df["at"]
    df["retained_earnings_ratio"] = df["re"] / df["at"]
    df["working_capital_ratio"] = (df["act"] - df["lct"]) / df["at"]

    df["dch_wc"] = df["working_capital_ratio"].diff()
    df["ch_rsst"] = (df["act"] - df["che"] - df["lct"] + df["dlc"]).diff() / df["at"]
    df["dch_rec"] = df["rect"].pct_change()
    df["dch_inv"] = df["invt"].pct_change()
    df["soft_assets"] = (df["at"] - df["rect"] - df["invt"] - df["che"]) / df["at"]
    df["ch_cs"] = df["sale"].pct_change() - df["rect"].pct_change()
    df["ch_cm"] = df["profit_margin"].diff()
    df["ch_roa"] = (df["ni"] / df["at"]).diff()
    df["issue"] = (df["dltt"].diff() > 0).astype(int)
    df["bm"] = df["ceq"] / (df["csho"] * df["prcc_f"])
    df["dpi"] = (df["cogs"] / df["dp"]).pct_change()
    df["reoa"] = df["ni"] / df["at"]
    df["ch_fcf"] = (df["ni"] - df["dp"]).pct_change()

    df["sales_growth"] = df["sale"].pct_change()
    df["inventory_growth"] = df["invt"].pct_change()
    df["receivable_growth"] = df["rect"].pct_change()
    df["asset_growth"] = df["at"].pct_change()
    df["debt_growth"] = df["dltt"].pct_change()
    df["margin_change"] = df["profit_margin"].diff()
    df["roa_change"] = df["ch_roa"].diff()

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def score_company(ticker):
    if model is None:
        return {"error": "Model weights not loaded. Train the model and save fraud.pt first."}

    try:
        df = fetch_financials(ticker)
    except Exception as e:
        return {"error": f"Could not fetch data for {ticker}: {str(e)}"}

    df = compute_derived(df)
    df_clean = df.dropna(subset=feature_cols)

    if len(df_clean) < seq_len:
        return {"error": f"Not enough clean data ({len(df_clean)} years available, need {seq_len})."}

    seq = df_clean[feature_cols].values[-seq_len:]
    seq_scaled = scaler.transform(seq)
    x = torch.tensor(seq_scaled[np.newaxis, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        fp, dp, cp, mp = model(x)
        return {
            "ticker": ticker.upper(),
            "raw": {
                "fraud": torch.sigmoid(fp).item(),
                "distress": torch.sigmoid(dp).item(),
                "credit": torch.sigmoid(cp).item(),
                "manip": torch.sigmoid(mp).item(),
            }
        }


def percentile_rank(score, distribution):
    return float(np.mean(np.array(distribution) <= score) * 100)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    result = score_company(ticker)
    if "error" in result:
        return jsonify(result), 400

    info = yf.Ticker(ticker).info
    sector = info.get("sector", "Unknown")
    name = info.get("longName", ticker)
    peers = peer_scores.get(sector, {})
    raw = result["raw"]

    percentiles = {}
    for key in ["fraud", "distress", "credit", "manip"]:
        percentiles[key] = percentile_rank(raw[key], peers[key]) if peers and key in peers else None

    return jsonify({
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "raw": raw,
        "percentiles": percentiles,
    })


@app.route("/health")
def health():
    return jsonify({"model_loaded": model is not None, "peer_sectors": list(peer_scores.keys())})


if __name__ == "__main__":
    app.run(debug=True, port=5000)