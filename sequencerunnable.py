from dotenv import load_dotenv
load_dotenv()  

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_template("Explain the {topic} in simple terms.")

model =ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

formatted_prompt = prompt.format_messages(topic="Machine Learning")

response = model.invoke(formatted_prompt)

final_output= parser.parse(response.content)

print(final_output)