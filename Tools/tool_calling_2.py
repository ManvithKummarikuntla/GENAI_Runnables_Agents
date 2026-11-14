from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.messages import HumanMessage
from rich import print

@tool
def get_text_length(text: str) -> int:
    """Return the number of characters from the given text"""
    return len(text)

tools = {
    "get_text_length":get_text_length
}

llm = ChatGroq(model="openai/gpt-oss-20b")
llm_with_tool = llm.bind_tools([get_text_length])

message=[]

while True:
    
    prompt=input("You : ")
    if prompt == "exit":
        break
    query=HumanMessage(prompt)
    message.append(query)

    result =llm_with_tool.invoke(message)
    message.append(result)

    if result.tool_calls:
        tool_name = result.tool_calls[0]["name"]
        tool_message=tools[tool_name].invoke(result.tool_calls[0])
        message.append(tool_message)


    result=llm_with_tool.invoke(message)
    print(result.content)



