import os
import json
import pickle
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from flask import Flask, render_template, request, Response, redirect, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from entmax import sparsemax

app = Flask(__name__)
app.secret_key = "finrisk-dev-secret-key"
app.permanent_session_lifetime = timedelta(days=365)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///finrisk.db"
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)


class WatchlistItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    ticker = db.Column(db.String(10), nullable=False)


with app.app_context():
    db.create_all()


@app.context_processor
def inject_user():
    current_user = None
    if "user_id" in session:
        current_user = User.query.get(session["user_id"])
    return {"current_user": current_user}


def fmt_large(n):
    if n is None:
        return "—"
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,.0f}"


def fmt_pct(n):
    if n is None:
        return "—"
    return f"{n * 100:.1f}%"


def fmt_num(n, decimals=2):
    if n is None:
        return "—"
    return f"{n:.{decimals}f}"


def tier_class(tier):
    return {"High": "danger", "Medium": "warning", "Low": "success", "Unknown": "secondary"}[tier]


app.jinja_env.filters["fmt_large"] = fmt_large
app.jinja_env.filters["fmt_pct"] = fmt_pct
app.jinja_env.filters["fmt_num"] = fmt_num
app.jinja_env.filters["tier_class"] = tier_class

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

non_temporal_required = [
    "act", "at", "ceq", "che", "cogs", "csho",
    "lct", "lt", "ni", "sale", "prcc_f", "EBIT",
    "current_ratio", "debt_ratio", "profit_margin", "ebit_margin",
    "asset_turnover", "retained_earnings_ratio", "working_capital_ratio",
    "soft_assets", "reoa", "bm",
]

seq_len = 5

info_fields = [
    "marketCap", "trailingPE", "totalRevenue", "fullTimeEmployees",
    "longBusinessSummary", "profitMargins", "returnOnAssets", "returnOnEquity",
    "debtToEquity", "currentRatio", "revenueGrowth", "grossMargins", "trailingEps",
    "heldPercentInstitutions",
]


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

model = fraud_model(len(feature_cols)).to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), "fraud.pt"),
    map_location=device,
    weights_only=True
))
model.eval()

with open(os.path.join(os.path.dirname(__file__), "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

peer_scores_file = os.path.join(os.path.dirname(__file__), "peer_scores.json")
peer_scores = {}
if os.path.exists(peer_scores_file):
    with open(peer_scores_file) as f:
        peer_scores = json.load(f)


def normalize_df(df):
    df = df.copy()
    df.columns = [
        str(pd.Timestamp(c).tz_convert(None))[:10]
        if pd.Timestamp(c).tzinfo is not None
        else str(pd.Timestamp(c))[:10]
        for c in df.columns
    ]
    return df


def fetch_financials(ticker):
    tk = yf.Ticker(ticker)
    bs = normalize_df(tk.balance_sheet)
    inc = normalize_df(tk.income_stmt)
    cf = normalize_df(tk.cashflow)
    info = tk.info

    years = sorted(bs.columns)
    rows = []

    for col in years:
        def g(src, *keys, _col=col):
            for k in keys:
                if k in src.index and _col in src.columns:
                    v = src.loc[k, _col]
                    return float(v) if pd.notna(v) else np.nan
            return np.nan

        def m(src, *keys):
            v = g(src, *keys)
            return v / 1e6 if not (v is None or np.isnan(v)) else np.nan

        def mz(src, *keys):
            v = m(src, *keys)
            return v if not (v is None or np.isnan(v)) else 0.0

        ni = m(inc, "Net Income")
        txt = mz(inc, "Tax Provision")
        xint = mz(inc, "Interest Expense")
        at_val = m(bs, "Total Assets")
        raw_ebit = (ni + txt + xint) if (ni is not None and not np.isnan(ni)) else np.nan
        ebit_val = (raw_ebit / at_val) if (
            raw_ebit is not None and not np.isnan(raw_ebit)
            and at_val is not None and not np.isnan(at_val)
            and at_val != 0
        ) else np.nan

        rows.append({
            "fyear": int(col[:4]),
            "act": m(bs, "Current Assets"),
            "ap": mz(bs, "Accounts Payable"),
            "at": at_val,
            "ceq": m(bs, "Stockholders Equity", "Total Equity Gross Minority Interest"),
            "che": m(bs, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
            "cogs": mz(inc, "Cost Of Revenue"),
            "csho": m(bs, "Share Issued", "Common Stock Shares Outstanding"),
            "dlc": mz(bs, "Current Debt", "Short Term Debt"),
            "dltis": mz(cf, "Issuance Of Debt"),
            "dltt": mz(bs, "Long Term Debt"),
            "dp": mz(cf, "Depreciation And Amortization", "Depreciation"),
            "ib": m(inc, "Net Income Common Stockholders", "Net Income"),
            "invt": mz(bs, "Inventory"),
            "ivao": mz(bs, "Other Investments", "Long Term Investments"),
            "ivst": mz(bs, "Short Term Investments"),
            "lct": m(bs, "Current Liabilities"),
            "lt": m(bs, "Total Liabilities Net Minority Interest", "Total Liabilities"),
            "ni": ni,
            "pstk": mz(bs, "Preferred Stock"),
            "re": mz(bs, "Retained Earnings"),
            "rect": mz(bs, "Receivables", "Net Receivables"),
            "sale": m(inc, "Total Revenue"),
            "sstk": mz(cf, "Common Stock Issuance", "Issuance Of Capital Stock"),
            "txp": mz(bs, "Taxes Payable", "Income Tax Payable"),
            "txt": txt,
            "xint": xint,
            "prcc_f": info.get("previousClose", np.nan),
            "EBIT": ebit_val,
            "ppegt": mz(bs, "Net PPE", "Properties", "Gross PPE"),
            "_raw_ebit": raw_ebit,
        })

    if not rows:
        return pd.DataFrame(columns=["fyear"] + feature_cols)

    return pd.DataFrame(rows).sort_values("fyear").reset_index(drop=True)


def compute_derived(df):
    df = df.copy()
    at_avg = (df["at"] + df["at"].shift(1)) / 2

    df["current_ratio"] = df["act"] / df["lct"]
    df["debt_ratio"] = (df["dlc"] + df["dltt"]) / df["at"]
    df["cash_ratio"] = df["che"] / df["lct"]
    df["profit_margin"] = df["ni"] / df["sale"]
    df["ebit_margin"] = df["_raw_ebit"] / df["sale"]
    df["asset_turnover"] = df["sale"] / df["at"]
    df["retained_earnings_ratio"] = df["re"] / df["at"]
    df["working_capital_ratio"] = (df["act"] - df["lct"]) / df["at"]

    wc = (df["act"] - df["che"]) - (df["lct"] - df["dlc"])
    nco = (df["at"] - df["act"] - df["ivao"]) - (df["lt"] - df["lct"] - df["dltt"])
    fin = (df["ivst"] + df["ivao"]) - (df["dltt"] + df["dlc"] + df["pstk"])

    df["ch_rsst"] = (wc.diff() + nco.diff() + fin.diff()) / at_avg
    df["dch_wc"] = wc.diff() / at_avg
    df["dch_rec"] = df["rect"].diff() / at_avg
    df["dch_inv"] = df["invt"].diff() / at_avg
    df["soft_assets"] = (df["at"] - df["ppegt"] - df["che"]) / df["at"]
    df["ch_cs"] = (df["sale"].diff() - df["rect"].diff()) / at_avg
    df["ch_cm"] = df["profit_margin"].diff()
    df["ch_roa"] = (df["ni"] / at_avg).diff()
    df["issue"] = (df["dltt"].diff() > 0).astype(int)
    df["bm"] = df["ceq"] / (df["csho"] * df["prcc_f"])
    df["dpi"] = (df["cogs"] / df["dp"]).pct_change()
    df["reoa"] = (df["sale"] - df["cogs"]) / df["at"]
    df["EBIT"] = df["_raw_ebit"] / at_avg
    df["ch_fcf"] = (df["ni"] - df["dp"]).diff() / at_avg

    df["sales_growth"] = df["sale"].pct_change()
    df["inventory_growth"] = df["invt"].pct_change()
    df["receivable_growth"] = df["rect"].pct_change()
    df["asset_growth"] = df["at"].pct_change()
    df["debt_growth"] = df["dltt"].pct_change()
    df["margin_change"] = df["profit_margin"].diff()
    df["roa_change"] = df["ch_roa"].diff()

    df = df.replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)
    df = df.drop(columns=["_raw_ebit", "ppegt"])
    df = df.dropna(subset=non_temporal_required)
    return df


min_seq_len = 1


def score_company(ticker):
    df = fetch_financials(ticker)
    df = compute_derived(df)

    if len(df) < min_seq_len:
        return {"error": f"Not enough data ({len(df)} years available, need at least {min_seq_len})."}

    seq = df[feature_cols].values[-seq_len:]
    seq_scaled = seq.copy()
    seq_scaled[:, :-7] = scaler.transform(pd.DataFrame(seq[:, :-7], columns=feature_cols[:-7]))
    x = torch.tensor(seq_scaled[np.newaxis, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        fp, dp, cp, mp = model(x)
        return {
            "ticker": ticker.upper(),
            "scores": {
                "fraud": torch.sigmoid(fp).item(),
                "distress": torch.sigmoid(dp).item(),
                "credit": torch.sigmoid(cp).item(),
                "manip": torch.sigmoid(mp).item(),
            }
        }


def percentile_rank(score, distribution):
    return float(np.mean(np.array(distribution) <= score) * 100)


def risk_tier(pctl):
    if pctl is None:
        return "Unknown"
    if pctl >= 90:
        return "High"
    if pctl >= 60:
        return "Medium"
    return "Low"


def build_tiers(scores, sector):
    peers = peer_scores.get(sector, {})
    tiers = {}
    for key in ["fraud", "distress", "credit", "manip"]:
        pctl = percentile_rank(scores[key], peers[key]) if peers and key in peers else None
        tiers[key] = risk_tier(pctl)
    return tiers


def extract_news_item(item):
    content = item.get("content", item)
    provider = content.get("provider", {})
    publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
    link_obj = content.get("canonicalUrl", {})
    link = link_obj.get("url", "") if isinstance(link_obj, dict) else ""
    return {
        "title": content.get("title", ""),
        "publisher": publisher or item.get("publisher", ""),
        "link": link or item.get("link", ""),
    }


def build_news(ticker):
    items = yf.Ticker(ticker).news or []
    parsed = [extract_news_item(item) for item in items[:5]]
    return [p for p in parsed if p["title"]]


def build_company_result(ticker):
    result = score_company(ticker)
    if "error" in result:
        return {"ticker": ticker, "error": result["error"]}

    info = yf.Ticker(ticker).info
    sector = info.get("sector", "Unknown")
    scores = result["scores"]

    tiers = build_tiers(scores, sector)

    return {
        "ticker": ticker,
        "name": info.get("longName", ticker),
        "sector": sector,
        "industry": info.get("industry", ""),
        "country": info.get("country", ""),
        "tiers": tiers,
        "scores": scores,
        "info": {k: info.get(k) for k in info_fields},
        "news": build_news(ticker),
    }


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/results")
def results_page():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return render_template("home.html")

    result = build_company_result(ticker)
    return render_template("results.html", result=result, ticker=ticker)


@app.route("/compare", methods=["GET", "POST"])
def compare_page():
    if request.method == "GET":
        return render_template("compare.html", results=None, submitted=[])

    tickers = [t.strip().upper() for t in request.form.getlist("tickers") if t.strip()]

    if not tickers:
        return render_template("compare.html", results=None, submitted=[])

    results = [build_company_result(ticker) for ticker in tickers]
    clean = sorted(
        (r for r in results if "error" not in r),
        key=lambda r: r["scores"]["fraud"],
        reverse=True,
    )
    errored = [r for r in results if "error" in r]

    return render_template("compare.html", results=clean + errored, submitted=tickers)


@app.route("/compare/export", methods=["POST"])
def compare_export():
    tickers = [t.strip().upper() for t in request.form.getlist("tickers") if t.strip()]
    results = [build_company_result(ticker) for ticker in tickers]

    lines = ["ticker,name,sector,fraud,distress,credit,manipulation"]
    for r in results:
        if "error" in r:
            lines.append(f"{r['ticker']},error,{r['error']},,,,")
            continue
        t = r["tiers"]
        lines.append(f"{r['ticker']},{r['name']},{r['sector']},{t['fraud']},{t['distress']},{t['credit']},{t['manip']}")

    csv_body = "\n".join(lines)
    return Response(
        csv_body,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=finrisk_comparison.csv"},
    )


@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "GET":
        return render_template("signup.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not username or not password:
        return render_template("signup.html", error="Username and password are required.")

    if User.query.filter_by(username=username).first():
        return render_template("signup.html", error="That username is already taken.")

    user = User(username=username, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()

    session.permanent = True
    session["user_id"] = user.id
    return redirect("/watchlist")


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        return render_template("login.html", error=None)

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password_hash, password):
        return render_template("login.html", error="Incorrect username or password.")

    session.permanent = True
    session["user_id"] = user.id
    return redirect("/watchlist")


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return redirect("/")


@app.route("/watchlist")
def watchlist_page():
    if "user_id" not in session:
        return redirect("/login")

    items = WatchlistItem.query.filter_by(user_id=session["user_id"]).all()
    results = [build_company_result(item.ticker) for item in items]
    return render_template("watchlist.html", results=results)


@app.route("/watchlist/add", methods=["POST"])
def watchlist_add():
    if "user_id" not in session:
        return redirect("/login")

    ticker = request.form.get("ticker", "").strip().upper()
    exists = WatchlistItem.query.filter_by(user_id=session["user_id"], ticker=ticker).first()
    if ticker and not exists:
        db.session.add(WatchlistItem(user_id=session["user_id"], ticker=ticker))
        db.session.commit()

    return redirect(request.referrer or "/watchlist")


@app.route("/watchlist/remove", methods=["POST"])
def watchlist_remove():
    if "user_id" not in session:
        return redirect("/login")

    ticker = request.form.get("ticker", "").strip().upper()
    WatchlistItem.query.filter_by(user_id=session["user_id"], ticker=ticker).delete()
    db.session.commit()

    return redirect(request.referrer or "/watchlist")


if __name__ == "__main__":
    app.run(debug=True)
