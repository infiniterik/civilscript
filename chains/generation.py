from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from dotenv import load_dotenv

from typing import Dict, List

load_dotenv()
llm = OpenAI(temperature=0.9)

stanceToTextPrompt = PromptTemplate(
    input_variables=["predicate", "belief_type", "belief", "sentiment"],
    template="""The author thinks that {predicate} is an expression of {belief_type} 
    and {belief} that this is true. The author feels {sentiment} about this fact."""
)

summarizationPrompt = PromptTemplate(
    input_variables=["stances", "domain"],
    template="""Write a comment that implicitly expresses the following stances on the topic of {domain}.
    Tell a story and make sure to avoid directly rephrasing the stances.
    Stances:
    {stances}""",
)

rewritePrompt = PromptTemplate(
    input_variables=["text"],
    template="""Rewrite the following text in your own words:
    {text}
    Rewritten:""",
)

rewriteChain = LLMChain(llm=llm, prompt=rewritePrompt, output_key="post")
summarizationChain = LLMChain(llm=llm, prompt=summarizationPrompt, output_key="post")

class Generation(Chain):
    @property
    def input_keys(self) -> List[str]:
        return ['stances', 'domain']

    @property
    def output_keys(self) -> List[str]:
        return ['post']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        stances = "\n".join([stanceToTextPrompt.format(**stance) for stance in inputs["stances"]])
        summary=summarizationChain.run(stances=stances, domain=inputs["domain"])
        return {"post": summary}

GenerationChain = Generation()