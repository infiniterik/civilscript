import requests, os
import dotenv
dotenv.load_dotenv()

from chains.stanceDrivenChains import explanationFromTextChain, explanationFromSymbolicStance, explanationFromNeuralStance, neuralStanceDetectionChain
from chains.symbolic import SymbolifyChain, symbolify

import streamlit as st

st.title("Stance Driven Explanation Generation Demos")

if "domains" not in st.session_state:
    st.session_state["domains"] = requests.get(f"{os.getenv('STANCE_SERVER')}/stance/domains").json()
    print(st.session_state["domains"])

domain = st.selectbox("Select a domain", st.session_state["domains"])

text = st.text_area("Enter a comment")

def getSymbol(text, explanation, domain):
    response = SymbolifyChain(dict(text=text, explanation=explanation, domain=domain))
    return f"<{response['predicate']}, {response['belief']}, {response['sentiment']}>"

if st.button("Generate Explanation"):
    st.write("Explanation from Text")
    explanation = explanationFromTextChain(dict(text=text, domain=domain))["explanation"]
    st.markdown(explanation)
    st.write("Symbolic Explanation")
    st.markdown(getSymbol(text, explanation, domain))

    st.write("Explanation from Symbolic Stance")
    explanation = explanationFromSymbolicStance(dict(text=text, domain=domain))["explanation"]
    st.markdown(explanation)
    st.write("Symbolic Explanation")
    st.markdown(getSymbol(text, explanation, domain))

    st.write("Explanation from Neural Stance")
    explanation = explanationFromNeuralStance(dict(text=text, domain=domain))["explanation"]
    st.markdown(explanation)
    st.write("Symbolic Explanation")
    st.markdown(getSymbol(text, explanation, domain))

    st.write("Stances from Neural Stance")
    explanation = neuralStanceDetectionChain(dict(text=text, domain=domain))["stances"]
    st.markdown(explanation)
    st.write("Symbolic Explanation")
    st.markdown(getSymbol(text, explanation, domain))