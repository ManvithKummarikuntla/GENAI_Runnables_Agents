from langchain.tools import tool


@tool
#to convert it to a tool we use tools
def get_greet(name : str) -> str:
    """Generate a greeting Message for a user !"""
    return f"Hello {name}, Welcome to the AI world!!"

#now we can use this tool with out llm
#tools are also called as runnables

result = get_greet.invoke({"name": "Manvith"})
print(result)

#to view the name of the tool

print(get_greet.name)
print(get_greet.description)
print(get_greet.args)