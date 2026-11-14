# City Intelligence System
### GenAI Agent | OpenWeather + Tavily + Groq

from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rich import print
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_OPEN_KEY")


# ==========================================
# 2. INITIALIZE API CLIENTS
# ==========================================

Tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ==========================================
# 3. WEATHER TOOL - OPENWEATHER API
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather information of a city."""

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": f"{city},IN",
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        data = response.json()

        if response.status_code != 200:
            return f"Weather Error: {data.get('message', 'Unable to fetch weather')}"

        # Extract weather information
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        return (
            f"Weather in {city}:\n"
            f"Condition: {description}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} m/s"
        )

    except requests.RequestException as e:
        return f"Weather API Error: {str(e)}"


# ==========================================
# 4. NEWS TOOL - TAVILY SEARCH API
# ==========================================

@tool
def get_news(city: str) -> str:
    """Search for the latest news and updates about a city."""

    try:
        response = Tavily_client.search(
            query=f"Latest news in {city}",
            search_depth="basic",
            max_results=3
        )

        results = response.get("results", [])

        if not results:
            return f"No news found for {city}."

        news_list = []

        for result in results:

            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("content", "")

            news_list.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Summary: {snippet[:300]}"
            )

        return (
            f"Latest News in {city}:\n\n"
            + "\n\n".join(news_list)
        )

    except Exception as e:
        return f"Tavily Search Error: {str(e)}"


# ==========================================
# 5. INITIALIZE GROQ LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ==========================================
# 6. REGISTER TOOLS
# ==========================================

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools(
    [get_weather, get_news]
)


# ==========================================
# 7. AGENT LOOP
# ==========================================

messages = []

print("\n[bold cyan]====================================[/bold cyan]")
print("[bold cyan]       CITY INTELLIGENCE SYSTEM     [/bold cyan]")
print("[bold cyan]====================================[/bold cyan]")

print("[green]Weather Intelligence[/green] | [yellow]Latest City News[/yellow]")
print("[dim]Powered by Groq + OpenWeather + Tavily[/dim]")

print("\n[bold]Type 'exit' to quit.[/bold]\n")


while True:

    user_input = input("[bold green]You:[/bold green] ")

    if user_input.lower().strip() == "exit":
        print("\n[bold yellow]Exiting City Intelligence System...[/bold yellow]")
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    while True:

        result = llm_with_tools.invoke(messages)

        messages.append(result)

        # ----------------------------------
        # Check whether the LLM requested tools
        # ----------------------------------

        if result.tool_calls:

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"]

                print(
                    f"\n[bold cyan]Agent requested:[/bold cyan] {tool_name}"
                )

                # Human-in-the-loop approval
                confirm = input(
                    f"Approve tool execution? (Y/N): "
                )

                if confirm.lower().strip() != "y":

                    tool_result = "Tool execution denied by the user."

                    print("[yellow]Tool execution denied.[/yellow]")

                else:

                    print(
                        f"[bold blue]Executing {tool_name}...[/bold blue]"
                    )

                    tool_result = tools[tool_name].invoke(
                        tool_call["args"]
                    )

                    print("[green]Tool execution completed.[/green]")

                # Send tool result back to the LLM
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

            continue

        # ----------------------------------
        # Display final LLM response
        # ----------------------------------

        else:

            print("\n[bold magenta]Assistant:[/bold magenta]")
            print(result.content)

            break
"""

## Project capabilities

- **Weather Intelligence:** Fetches current weather using OpenWeather.
- **News Intelligence:** Searches current web information using Tavily.
- **LLM Reasoning:** Groq determines which tools are required.
- **Tool Calling:** LangChain connects Python functions to the LLM.
- **Human-in-the-Loop:** User approves or denies tool execution.
- **Conversation Context:** Maintains message history during the session.

"""