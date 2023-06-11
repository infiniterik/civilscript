# Symbolic Stance Detection

import requests
from langchain.chains.base import Chain

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

stanceDescription = """A stance is a combination of a predicate expressed by the author, whether or not the author believes said predicate, and the author's sentiment towards the predicate."""

import json
import sys, os
from typing import Dict, List

llm = OpenAI(temperature=0.9)

getPredicate = PromptTemplate(
    input_variables=["text", "explanation", "domain"],
    template=stanceDescription+"""
    Consider the following comment and explanation regarding stances about {domain} the text expresses. 
    What is the main predicate that stances refer to?
    Comment:{text}
    Explanation: {explanation}
    Predicate:""",
)

getPredicateChain = LLMChain(llm=llm, prompt=getPredicate, output_key="predicate")

getSentiment = PromptTemplate(
    input_variables=["text", "predicate", "explanation", "domain"],
    template=stanceDescription+"""
    Consider the following comment and explanation regarding stances about {domain} the text expresses. 
    What is the sentiment of the author towards the predicate {predicate}? Respond with one of the following:
    - Positive
    - Negative
    - Neutral
    Comment:{text}
    Explanation: {explanation}
    Sentiment:""",
)

getSentimentChain = LLMChain(llm=llm, prompt=getSentiment, output_key="sentiment")

getBelief = PromptTemplate(
    input_variables=["text", "predicate", "explanation", "domain"],
    template=stanceDescription+"""
    Consider the following comment and explanation regarding stances about {domain} the text expresses. 
    How strongly does the author believe the predicate {predicate}? Respond with one of the following:
    - Very strongly believes
    - Strongly believes
    - Believes
    - Does not believe
    - Strongly does not believe
    - Very strongly does not believe

    Comment:{text}
    Explanation: {explanation}
    Sentiment:""",
)

getBeliefChain = LLMChain(llm=llm, prompt=getBelief, output_key="belief")


SymbolifyChain = SequentialChain(chains=[getPredicateChain, getSentimentChain, getBeliefChain],
                                                input_variables=["text", "explanation", "domain"],
                                                output_variables=["predicate", "sentiment", "belief"],
                                                verbose=True)