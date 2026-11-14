from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough


model = ChatGroq(model="openai/gpt-oss-20b")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system" ,"You are a code generator."),
    ("human" , "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant who explains code in simple"),
    ("human","Explain the following code in simple words :\n{code}")
])

seq = code_prompt | model | parser
seq2 = RunnableParallel(
    {"code":RunnablePassthrough(),
     "explaination" : explain_prompt | model | parser
     }

)

chain = seq | seq2


result = chain.invoke({"topic": "Write a code of palindorome in python"})

print(result['code'])
print(result['explaination'])
