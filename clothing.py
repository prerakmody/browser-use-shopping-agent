import asyncio
import streamlit as st
from dotenv import load_dotenv
from browser_use import Agent
from langchain_openai import ChatOpenAI

load_dotenv()

async def run_agent(task):
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    return result

def main():
    st.title("Clothing Search App")

    query = st.text_input("Enter your query:")
    size = st.selectbox("Select size:", ["S", "M", "L"])
    websites = st.multiselect("Select websites to query:", ["zalando.nl", "hm.com", "zara.nl"])
    result_count = st.slider("Number of results:", 5, 15, 5)

    if st.button("Search"):
        if query and websites:
            task = f"Search for '{query}' of size {size} on {', '.join(websites)}. Return a .json of the image URLs, site name, price, and other clothing properties for the top {result_count} results."
            result = asyncio.run(run_agent(task))
            st.json(result)
        else:
            st.error("Please enter a query and select at least one website.")

if __name__ == "__main__":
    main()