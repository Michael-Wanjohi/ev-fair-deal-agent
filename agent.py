"""
agent.py - Used-EV Fair Deal Agent (Team Green prototype)

Steps the agent runs for every request:
  1. scope_check     - refuse out-of-scope requests (loans, credit, legal advice)
  2. parse_listing   - read pasted listing text -> fields (LLM if key, else simple rules)
  3. check_input     - are the fields complete and believable?
  4. predict_price   - ML tool: ridge + gradient boosting (10/50/90%) -> fair range
  5. find_comps      - real past resales of the same car
  6. risk_flags      - when should a human look at it?
  7. policy_lookup   - small incentive knowledge base (verified facts only)
  8. decide          - GOOD BUY / FAIR - NEGOTIATE / OVERPRICED / ESCALATE
  9. write_note      - plain-English note (LLM if key, else template)
 10. verify_note     - every $ number in the note must come from the tools
Each step is saved in a trace so we can show and test the agent's path.
"""
import os
import re
import json
import numpy as np
import pandas as pd
import joblib
from train_models import add_features

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = joblib.load(os.path.join(HERE, "price_models.pkl"))
DATA = add_features(pd.read_csv(os.path.join(HERE, "used_resales_clean.csv")))
KNOWN_MAKES = sorted(DATA["make"].unique())
KNOWN_MODELS = DATA.groupby("make")["model"].unique().to_dict()

# decision rules (agreed with the team, can be tuned in sensitivity analysis)
GOOD_BUY = 0.90      # asking <= 90% of typical price
OVERPRICED = 1.10    # asking > 110% of typical price
MIN_COMPS = 30
WIDE_RANGE = 0.60    # (high - low) / typical
MODEL_DISAGREE = 0.15

OUT_OF_SCOPE = ["loan", "credit score", "credit decision", "approve", "financing", "finance",
                "interest rate", "lawsuit", "legal advice", "insurance claim", "personal data", "ssn"]


# ---------------- optional LLM (Claude API) ----------------
def llm(prompt, max_tokens=400):
    """Returns text from Claude, or None if no API key / package (agent still works)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                                     max_tokens=max_tokens,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    except Exception as e:
        print("LLM call failed, using fallback:", e)
        return None


# ---------------- tools ----------------
def scope_check(text):
    t = text.lower()
    hits = [w for w in OUT_OF_SCOPE if w in t]
    return {"in_scope": not hits, "matched": hits}


def parse_listing(text):
    """Pull make, model, model_year, odometer, asking price out of listing text."""
    out = None
    reply = llm("Extract fields from this used car listing. Reply with JSON only, keys: "
                "make, model, model_year, odometer, asking_price (numbers as numbers, null if missing). "
                f"Allowed makes: {KNOWN_MAKES}.\n\nListing:\n{text}")
    if reply:
        try:
            out = json.loads(reply[reply.find("{"): reply.rfind("}") + 1])
            out["parser"] = "LLM"
        except Exception:
            out = None
    if out is None:  # simple rule-based fallback
        up = text.upper()
        out = {"make": None, "model": None, "model_year": None, "odometer": None,
               "asking_price": None, "parser": "rules"}
        for mk in KNOWN_MAKES:
            if mk in up:
                out["make"] = mk
                for md in sorted(KNOWN_MODELS[mk], key=len, reverse=True):
                    if md in up:
                        out["model"] = md
                        break
        yr = re.search(r"\b(20[0-2]\d)\b", text)
        if yr:
            out["model_year"] = int(yr.group(1))
        mi = re.search(r"([\d,\.]+)\s*(k)?\s*(miles|mi)\b", text, re.I)
        if mi:
            v = float(mi.group(1).replace(",", ""))
            out["odometer"] = v * 1000 if mi.group(2) else v
        pr = re.search(r"\$\s*([\d,]+)", text)
        if pr:
            out["asking_price"] = float(pr.group(1).replace(",", ""))
    if out.get("make"):
        out["make"] = str(out["make"]).upper()
    if out.get("model"):
        out["model"] = str(out["model"]).upper()
    return out


def check_input(car, today):
    problems = []
    for k in ["make", "model", "model_year", "odometer", "asking_price"]:
        if car.get(k) in (None, ""):
            problems.append(f"missing {k}")
    if not problems:
        if car["make"] not in KNOWN_MAKES or car["model"] not in KNOWN_MODELS.get(car["make"], []):
            problems.append("car not in our data")
        if not (2010 <= car["model_year"] <= today.year + 1):
            problems.append("model year out of range")
        if not (0 <= car["odometer"] <= 300000):
            problems.append("odometer out of range")
        if not (1000 <= car["asking_price"] <= 250000):
            problems.append("asking price out of range")
    return {"ok": not problems, "problems": problems}


def make_row(car, today):
    m = DATA[(DATA.make == car["make"]) & (DATA.model == car["model"])]
    age = max(today.year - car["model_year"], 0)
    return pd.DataFrame([{
        "make": car["make"], "model": car["model"], "fuel": m["fuel"].mode()[0],
        "age": age, "odometer": car["odometer"], "miles_per_year": car["odometer"] / max(age, 1),
        "original_price": car.get("original_price", np.nan),
        "orig_missing": int(pd.isna(car.get("original_price", np.nan))),
        "month_index": (today.year - 2019) * 12 + today.month}])


def predict_price(car, today):
    X = make_row(car, today)[MODELS["features"]]
    low, mid, high = (float(np.exp(MODELS[k].predict(X)[0])) for k in ["low", "mid", "high"])
    ridge = float(np.exp(MODELS["ridge"].predict(X)[0]))
    low, high = min(low, mid), max(high, mid)
    return {"low": round(low, -2), "typical": round(mid, -2), "high": round(high, -2),
            "ridge_check": round(ridge, -2)}


def find_comps(car, today):
    m = DATA[(DATA.make == car["make"]) & (DATA.model == car["model"]) &
             (DATA.model_year.between(car["model_year"] - 1, car["model_year"] + 1)) &
             (DATA.sale_date >= today - pd.DateOffset(months=24)) & (DATA.sale_date <= today)]
    m = m.assign(gap=(m.odometer - car["odometer"]).abs()).sort_values("gap")
    return {"n": len(m),
            "median_price": float(m.price.median()) if len(m) else None,
            "examples": m.head(5)[["sale_date", "model_year", "odometer", "price"]]
                         .assign(sale_date=lambda d: d.sale_date.dt.date.astype(str)).to_dict("records")}


def risk_flags(car, price, comps, today):
    flags = []
    if comps["n"] < MIN_COMPS:
        flags.append(f"only {comps['n']} similar past sales (need {MIN_COMPS})")
    if (price["high"] - price["low"]) / price["typical"] > WIDE_RANGE:
        flags.append("price range is very wide")
    if abs(price["ridge_check"] - price["typical"]) / price["typical"] > MODEL_DISAGREE:
        flags.append("the two models disagree by more than 15%")
    age = max(today.year - car["model_year"], 1)
    mpy = car["odometer"] / age
    if mpy > 25000 or car["odometer"] < 100:
        flags.append("unusual mileage")
    if car["asking_price"] < 0.5 * price["typical"]:
        flags.append("price far below market - possible scam or damage")
    return flags


def policy_lookup(car, today):
    """Small knowledge base in policies/*.md. Only returns rules that apply to this date."""
    notes = []
    with open(os.path.join(HERE, "policies", "incentives.json")) as f:
        rules = json.load(f)
    for r in rules:
        start, end = pd.Timestamp(r["start"]), pd.Timestamp(r["end"]) if r["end"] else None
        if start <= today and (end is None or today <= end):
            notes.append(r["text"] + f" (source: {r['source']})")
        elif end is not None and today > end and r.get("show_if_ended"):
            notes.append("Ended: " + r["text"] + f" (ended {r['end']}; source: {r['source']})")
    return notes


def decide(car, price, flags):
    ratio = car["asking_price"] / price["typical"]
    if flags:
        label = "ESCALATE - ask a person to check"
    elif ratio <= GOOD_BUY:
        label = "GOOD BUY"
    elif ratio <= OVERPRICED:
        label = "FAIR - NEGOTIATE"
    else:
        label = "OVERPRICED"
    return {"label": label, "ratio": round(ratio, 3)}


def money(x):
    return f"${x:,.0f}"


def write_note(car, price, comps, decision, flags, policies):
    facts = {"car": f"{car['model_year']} {car['make']} {car['model']}, {car['odometer']:,.0f} miles",
             "asking": money(car["asking_price"]), "typical": money(price["typical"]),
             "range": f"{money(price['low'])} - {money(price['high'])}",
             "similar_sales": comps["n"], "decision": decision["label"], "flags": flags,
             "policies": policies}
    note = llm("Write a short, plain-English note (max 90 words) for a used-EV buyer. "
               "Use ONLY these facts and numbers, do not add any new numbers or advice "
               f"about loans or credit:\n{json.dumps(facts)}")
    if note is None:
        note = (f"{facts['car']}. Asking {facts['asking']}. Similar cars sold for about "
                f"{facts['typical']} (usual range {facts['range']}, based on {comps['n']} past sales "
                f"in Washington). Result: {decision['label']}.")
        if flags:
            note += " Please have a person check: " + "; ".join(flags) + "."
    return note, facts


def verify_note(note, car, price, comps):
    """Guardrail: every dollar amount in the note must match a tool output."""
    allowed = {round(v, -2) for v in [car["asking_price"], price["low"], price["typical"], price["high"]]}
    if comps["median_price"]:
        allowed.add(round(comps["median_price"], -2))
    found = [float(x.replace(",", "")) for x in re.findall(r"\$\s?([\d,]+)", note)]
    bad = [x for x in found if round(x, -2) not in allowed]
    return {"ok": not bad, "unknown_numbers": bad}


# ---------------- the agent loop ----------------
def run_agent(text=None, car=None, today=None):
    today = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    trace = []

    def step(name, result):
        trace.append({"step": name, "result": result})
        return result

    if text is not None:
        sc = step("scope_check", scope_check(text))
        if not sc["in_scope"]:
            return {"decision": "REFUSED", "note": "Sorry, I can only help with used-EV prices. "
                    "I can't help with loans, credit or legal questions.", "trace": trace}
        car = step("parse_listing", parse_listing(text))

    chk = step("check_input", check_input(car, today))
    if not chk["ok"]:
        return {"decision": "NEED MORE INFO", "note": "I need: " + ", ".join(chk["problems"]) + ".",
                "car": car, "trace": trace}

    price = step("predict_price", predict_price(car, today))
    comps = step("find_comps", find_comps(car, today))
    flags = step("risk_flags", risk_flags(car, price, comps, today))
    policies = step("policy_lookup", policy_lookup(car, today))
    decision = step("decide", decide(car, price, flags))
    note, facts = write_note(car, price, comps, decision, flags, policies)
    ver = step("verify_note", verify_note(note, car, price, comps))
    if not ver["ok"]:  # LLM added a number -> use the safe template instead
        os.environ.pop("ANTHROPIC_API_KEY", None)
        note, facts = write_note(car, price, comps, decision, flags, policies)
        step("verify_note_retry", verify_note(note, car, price, comps))
    return {"decision": decision["label"], "note": note, "car": car, "price": price,
            "comps": comps, "flags": flags, "policies": policies, "trace": trace}


if __name__ == "__main__":
    demo = "2021 Tesla Model 3, 38,000 miles, one owner, asking $24,500. Seattle."
    r = run_agent(text=demo, today="2026-06-15")
    print(r["decision"]); print(r["note"])
    for t in r["trace"]:
        print("-", t["step"], ":", str(t["result"])[:120])
