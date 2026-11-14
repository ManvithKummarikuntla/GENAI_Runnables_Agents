from dotenv import load_dotenv
load_dotenv()

import os 
import requests
#to hit any url online we use requests
from rich import print


from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient
from rich import print

#now lets create some tools

# weather tool

@tool
def get_weather(city : str) -> str:
    """Get current weather of a city"""
    API_KEY=os.getenv("OPENWEATHER_API_KEY")
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"


    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if str(data.get("cod")) !="200":
        return f"Error: {data.get('message',"Could not fetch weather")}"

    temp =data["main"]["temp"]
    desc= data["weather"][0]["description"]

    return f"Weather in {city} : {desc},{temp}C"

# print(get_weather.invoke("Khammam"))



#Tavily News tool

Tavily_client=TavilyClient(api_key=os.getenv("TAVILY_OPEN_KEY"))

@tool
def get_news(city : str) ->str:
    """Get the latest news about the city"""
    response = Tavily_client.search(
        query = f"latest news in {city}",
        search_depth = "basic",
        max_results=3
    )
    results = response.get("results",[])
    if not results:
        return f"No news found for {city}"
    news_list=[]

    for r in results:
        title = r.get("title","No title")
        url = r.get("ur;","")
        snippet = r.get("content","")
        news_list.append(
            f"- {title}\n {url}\n {snippet[:]}....."
            
        )
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

#print(get_news.invoke("Khammam"))

#LLM

llm = ChatGroq(model = "openai/gpt-oss-20b")

tools ={
    "get_weather":get_weather,
    "get_news":get_news
}
llm_with_tools = llm.bind_tools([get_weather,get_news])

#AGENT LOOP - VERY IMPORTANT

messages = []
print("City Intelligence System")
print("Type Exit to Quit")

while True:
    user_input= input("You : ")
    if user_input.lower()=="exit":
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        #if tool is required

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']


                #HUMAN in the loop

                confirm = input(f"Agent wants to call {tool_name} Approve(Y/N)")

                if confirm.lower() == "n":
                    print("tool call denied and i cannot get the latest information")

                    break
                #to execute tool
                tool_result=tools[tool_name].invoke(tool_call)

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id = tool_call['id']
                ))
            continue
        else:
            
            print(result.content)    



        
