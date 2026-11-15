
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

st.set_page_config(
    page_title="City AI Assistant",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 City AI Assistant")
st.caption("Your AI-powered assistant for weather and latest city news.")


# ==========================================
# 2. API CLIENTS
# ==========================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================================
# 3. WEATHER TOOL
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    if not OPENWEATHER_API_KEY:
        return "OpenWeather API key is missing."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if response.status_code != 200:
            return f"Weather error: {data.get('message', 'Unable to fetch weather')}"

        temperature = data["main"]["temp"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]

        return (
            f"Weather in {city}:\n"
            f"Temperature: {temperature}°C\n"
            f"Condition: {description}\n"
            f"Humidity: {humidity}%"
        )

    except requests.RequestException as e:
        return f"Weather API connection error: {e}"


# ==========================================
# 4. TAVILY NEWS TOOL
# ==========================================

tavily_client = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY else None
)


@tool
def get_news(city: str) -> str:
    """Search for the latest news about a city."""

    if not tavily_client:
        return "Tavily API key is missing."

    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
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
            f"Latest news about {city}:\n\n"
            + "\n\n".join(news_list)
        )

    except Exception as e:
        return f"News search error: {e}"


# ==========================================
# 5. HUMAN APPROVAL MIDDLEWARE
# ==========================================

@wrap_tool_call
def human_approval(request, handler):
    """Allow tool execution only when approved in Streamlit."""

    approved = st.session_state.get("approve_tools", False)

    if not approved:
        return ToolMessage(
            content="Tool execution was not approved by the user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


# ==========================================
# 6. CREATE AI AGENT
# ==========================================

@st.cache_resource
def create_city_agent():

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in .env")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather, get_news],
        system_prompt="""
        You are a helpful City AI Assistant.

        You can:
        1. Get current weather using get_weather.
        2. Find latest news using get_news.

        Understand the user's question and select the
        appropriate tool.

        If the user asks about both weather and news,
        use both tools.

        Present results clearly and conversationally.
        Do not invent weather or news information.
        """,
        middleware=[human_approval]
    )

    return agent


# ==========================================
# 7. STREAMLIT SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "approve_tools" not in st.session_state:
    st.session_state.approve_tools = False


# ==========================================
# 8. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.checkbox(
        "Allow AI tool execution",
        key="approve_tools",
        help="Enable this to let the agent call Weather and News APIs."
    )

    st.divider()

    st.subheader("Available Tools")

    st.write("🌤️ OpenWeather API")
    st.write("📰 Tavily News Search")
    st.write("🤖 Groq LLM")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# 9. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# 10. CHAT INPUT AND AGENT EXECUTION
# ==========================================

user_input = st.chat_input(
    "Ask about weather or news in any city..."
)

if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):

        try:
            with st.spinner("🤖 Thinking..."):

                agent = create_city_agent()

                # Pass previous conversation as context
                result = agent.invoke({
                    "messages": st.session_state.messages
                })

                response = result["messages"][-1].content

            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:

            error_message = f"Something went wrong: {e}"

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
