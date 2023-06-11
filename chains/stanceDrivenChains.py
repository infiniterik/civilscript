from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain



from dotenv import load_dotenv

from chains import symbolic

load_dotenv()

llm = OpenAI(temperature=0.9)

stanceDescription = """A stance is a combination of a predicate expressed by the author, whether or not the author believes said predicate, and the author's sentiment towards the predicate."""

# Symbolic Stance Detection with Neural Explanation
explanationPrompt = PromptTemplate(
    input_variables=["text", "stances"],
    template="""Write a short explanation explaining why the comment below evokes the following stances. Make sure not to add any hypotheses beyond what can be inferred directly from the text:
    Comment:{text}
    {stances}
    Explanation:""",
)

explanationFromTextAndStanceChain = LLMChain(llm=llm, prompt=explanationPrompt, output_key="explanation")

explanationFromSymbolicStance = SequentialChain(chains=[symbolic.StanceGetterChain, explanationFromTextAndStanceChain],
                                                input_variables=["text", "domain"],
                                                output_variables=["explanation", "stances", "representations"],
                                                verbose=True)

# Neural Stance Detection

stanceDetectionPrompt = PromptTemplate(
    input_variables=["text", "domain"],
    template="""Write a short explanation explaining why the comment below evokes the following stances about {domain}. Make sure not to add any hypotheses beyond what can be inferred directly from the text:
    Comment:{text}
    Stances:""",
)

neuralStanceDetectionChain = LLMChain(llm=llm, prompt=stanceDetectionPrompt, output_key="stances")

# Neural Stance Detection with Neural Explanation

explanationFromNeuralStance = SequentialChain(chains=[neuralStanceDetectionChain, explanationFromTextAndStanceChain],
                                                input_variables=["text", "domain"],
                                                output_variables=["explanation", "stances"],
                                                verbose=True)

# Neural Stance Description without Stance Detection
explanationWithoutStancesPrompt = PromptTemplate(
    input_variables=["text", "domain"],
    template=stanceDescription+"""
    Write a short explanation explaining why the comment below evokes stances about {domain}. 
    Be specific and respond in bullet point form but make sure not to add any hypotheses beyond what can be inferred directly from the text:
    Comment:{text}
    Explanation:""",
)

explanationFromTextChain = LLMChain(llm=llm, prompt=explanationWithoutStancesPrompt, output_key="explanation")