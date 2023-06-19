import requests, os
import dotenv

dotenv.load_dotenv()

from chains.stanceDrivenChains import explanationFromTextChain, explanationFromSymbolicStance, explanationFromNeuralStance, neuralStanceDetectionChain
from chains.symbolic import SymbolifyChain, SymbolifyWithoutExplanationChain
from chains.generation import GenerationChain, rewriteChain

import streamlit as st

st.title("Stance Driven Explanation Generation Demos")

if "domains" not in st.session_state:
    st.session_state["domains"] = requests.get(f"{os.getenv('STANCE_SERVER')}/stance/domains").json()
    print(st.session_state["domains"])

domain = st.selectbox("Select a domain", st.session_state["domains"])

text = st.text_area("Enter a comment")

def getStances(text, explanation, domain):
    return SymbolifyChain(dict(text=text, explanation=explanation, domain=domain))["stances"]

def getSymbols(stances):
    return "* " + "\n* ".join([f"`<{response['belief_type']}[{response['predicate']}], {response['belief']}, {response['sentiment']}>`" for response in stances])

if st.button("Generate Explanation"):
    st.header("Example")
    st.markdown("**Original Text:** "+text)
    st.markdown("**Domain:** " + domain + "\n\n")
    
    st.subheader("Rewrite directly from text")
    st.write(rewriteChain(dict(text=text))["post"])

    st.subheader("Explanation from Text")
    explanation = explanationFromTextChain(dict(text=text, domain=domain))["explanation"]
    st.markdown(explanation)
    st.write("**Symbolic Explanation**")
    stances = getStances(text, explanation, domain)
    st.markdown(getSymbols(stances))
    st.markdown("**Regenerated Explanation**")
    st.write(GenerationChain(dict(stances=stances, domain=domain))["post"])

    st.subheader("Explanation from Symbolic Stance")
    explanation = explanationFromSymbolicStance(dict(text=text, domain=domain))["explanation"]
    st.markdown(explanation)
    st.write("**Symbolic Explanation**")
    stances = getStances(text, explanation, domain)
    st.markdown(getSymbols(stances))
    st.markdown("**Regenerated Explanation**")
    st.write(GenerationChain(dict(stances=stances, domain=domain))["post"])

    st.subheader("Explanation from Neural Stance")
    explanation = explanationFromNeuralStance(dict(text=text, domain=domain))["explanation"]
    st.markdown(explanation)
    st.write("**Symbolic Explanation**")
    stances = getStances(text, explanation, domain)
    st.markdown(getSymbols(stances))
    st.markdown("**Regenerated Explanation**")
    st.write(GenerationChain(dict(stances=stances, domain=domain))["post"])

    st.subheader("Directly from Neural Stance")
    st.write("**Symbolic Explanation**")
    stances = SymbolifyWithoutExplanationChain(dict(text=text, domain=domain))["stances"]
    st.markdown(getSymbols(stances))
    st.markdown("**Regenerated Explanation**")
    st.write(GenerationChain(dict(stances=stances, domain=domain))["post"])
