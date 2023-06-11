import requests, os
import dotenv
dotenv.load_dotenv()

from chains.stanceDrivenChains import explanationFromTextChain, explanationFromSymbolicStance, explanationFromNeuralStance, neuralStanceDetectionChain

import streamlit as st

st.title("Stance Driven Explanation Generation Demos")

if "domains" not in st.session_state:
    st.session_state["domains"] = requests.get(f"{os.getenv('STANCE_SERVER')}/stance/domains").json()
    print(st.session_state["domains"])

domain = st.selectbox("Select a domain", st.session_state["domains"])

text = st.text_area("Enter a comment")

if st.button("Generate Explanation"):
    st.write("Explanation from Text")
    st.markdown(explanationFromTextChain(dict(text=text, domain=domain))["explanation"])
    st.write("Explanation from Symbolic Stance")
    st.markdown(explanationFromSymbolicStance(dict(text=text, domain=domain))["explanation"])
    st.write("Explanation from Neural Stance")
    st.markdown(explanationFromNeuralStance(dict(text=text, domain=domain))["explanation"])
    st.write("Stances from Neural Stance")
    st.markdown(neuralStanceDetectionChain(dict(text=text, domain=domain))["stances"])