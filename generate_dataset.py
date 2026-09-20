"""
generate_dataset.py
Builds a SIMULATED dataset of automated phone system (IVR) call sessions.
No real customer data is used. The relationships between columns
are assumptions I coded in myself so the dataset behaves like a realistic
call-center dataset (longer calls with more re-prompts, transfers, etc.).
Run:  python generate_dataset.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(634)
N = 1200

# ---- dates: Jan 1 - Mar 31, 2026, sorted so the data is time ordered ----
dates = pd.date_range("2026-01-01", "2026-03-31", freq="D")
day_idx = rng.integers(0, len(dates), N)
hour = rng.choice(np.arange(6, 23), N)           # hour of day is independent of everything else
df = pd.DataFrame({"call_date": dates[day_idx], "hour_of_day": hour})
df = df.sort_values(["call_date", "hour_of_day"]).reset_index(drop=True)

# ---- platform: share of calls on the new platform grows over the quarter ----
progress = (df["call_date"] - dates[0]).dt.days / (len(dates) - 1)
p_new = 0.15 + 0.65 * progress
df["platform"] = np.where(rng.random(N) < p_new, "New Platform", "Legacy IVR")
new = (df["platform"] == "New Platform").to_numpy()

# ---- intent ----
intents = ["Make a Payment", "Recent Transactions", "Balance Inquiry",
           "Rewards Redemption", "Cancel a Payment"]
df["intent"] = rng.choice(intents, N, p=[0.32, 0.24, 0.18, 0.14, 0.12])
base = df["intent"].map({"Make a Payment": 200, "Recent Transactions": 140,
                         "Balance Inquiry": 90, "Rewards Redemption": 250,
                         "Cancel a Payment": 220}).to_numpy()

# ---- speech recognition, re-prompts, latency, turns ----
df["num_reprompts"] = rng.poisson(np.where(new, 0.55, 1.15))
asr = np.where(new, rng.beta(17, 2.4, N), rng.beta(13, 3.2, N))
df["asr_confidence"] = np.clip(asr, 0.30, 0.99).round(3)
latency = rng.lognormal(mean=np.where(new, 5.25, 5.45), sigma=0.28, size=N)
spike = rng.choice(N, 16, replace=False)         # a few latency spikes (slow backend calls)
latency[spike] *= rng.uniform(6, 10, 16)
df["api_latency_ms"] = latency.round(0)
df["num_turns"] = 3 + rng.poisson(base / 60) + df["num_reprompts"]

# ---- transfers to a live agent ----
p_transfer = 0.04 + 0.07 * df["num_reprompts"] + 0.35 * (df["asr_confidence"] < 0.65)
df["transferred_to_agent"] = (rng.random(N) < p_transfer.clip(0, 0.9)).astype(int)

# ---- call duration (seconds) ----
dur = (base * rng.lognormal(0, 0.22, N)
       + 22 * df["num_reprompts"] + 7 * df["num_turns"]
       + 0.05 * df["api_latency_ms"] + 110 * df["transferred_to_agent"])
dur = np.where(new, dur * 0.86, dur)
long_calls = rng.choice(N, 14, replace=False)    # stuck / abandoned sessions
dur[long_calls] = rng.uniform(1500, 3400, 14)
df["call_duration_sec"] = dur.round(0)

# ---- post-call survey score (1-5) ----
csat = (4.3 - 0.35 * df["num_reprompts"] - 0.9 * df["transferred_to_agent"]
        + rng.normal(0, 0.6, N))
df["csat_score"] = np.clip(np.round(csat), 1, 5)

# ---- call id ----
df.insert(0, "call_id", ["C" + str(10000 + i) for i in range(N)])

# ---- inject missing values ----
def blank(col, frac):
    idx = rng.choice(N, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

blank("asr_confidence", 0.06)
blank("api_latency_ms", 0.05)
blank("csat_score", 0.10)          # survey non-response
blank("intent", 0.015)
blank("call_duration_sec", 0.01)

cols = ["call_id", "call_date", "hour_of_day", "platform", "intent",
        "call_duration_sec", "num_turns", "num_reprompts", "asr_confidence",
        "api_latency_ms", "transferred_to_agent", "csat_score"]
df[cols].to_csv("data/ivr_call_sessions_raw.csv", index=False)
print(df[cols].shape)
