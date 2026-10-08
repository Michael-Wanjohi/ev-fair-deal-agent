"""
app.py - simple web screen for the agent.   Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
import agent

st.set_page_config(page_title="Used-EV Fair Deal Agent", page_icon="🔌")
st.title("Used-EV Fair Deal Agent")
st.caption("Team Green prototype - Washington used EVs. Prices only, no loan or credit advice.")

text = st.text_area("Paste the car listing",
                    "2021 Tesla Model 3, 38,000 miles, one owner, asking $24,500. Seattle.")
today = st.date_input("Date of check", pd.Timestamp("2026-06-15"))

if st.button("Check this deal"):
    r = agent.run_agent(text=text, today=today)
    color = {"GOOD BUY": "green", "FAIR - NEGOTIATE": "orange", "OVERPRICED": "red"}.get(r["decision"], "gray")
    st.markdown(f"### :{color}[{r['decision']}]")
    st.write(r["note"])
    if "price" in r:
        c1, c2, c3 = st.columns(3)
        c1.metric("Low (10%)", f"${r['price']['low']:,.0f}")
        c2.metric("Typical", f"${r['price']['typical']:,.0f}")
        c3.metric("High (90%)", f"${r['price']['high']:,.0f}")
        st.subheader("Similar past sales")
        st.dataframe(pd.DataFrame(r["comps"]["examples"]))
        if r["policies"]:
            st.subheader("Incentives check")
            for p in r["policies"]:
                st.write("- " + p)
    with st.expander("How the agent decided (step trace)"):
        for t in r["trace"]:
            st.write(f"**{t['step']}**", t["result"])
