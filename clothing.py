import asyncio
import streamlit as st
from dotenv import load_dotenv
from browser_use import Agent
from langchain_openai import ChatOpenAI

load_dotenv()

async def run_agent(task):
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o", timeout=60000), # api_key="", model="gpt-3.5-turbo", timeout=60000 (ms)
    )
    result = await agent.run()
    return result

def main():
    st.title("Clothing Search App")

    query = st.text_input("Enter your query:")
    size = st.selectbox("Select size:", ["S", "M", "L"])
    sex = st.radio("Select sex:", ["Male", "Female", "Unisex"])
    websites = st.multiselect("Select websites to query:", ["zalando.nl", "hm.com", "zara.nl"])
    result_count = st.slider("Number of results:", 5, 15, 5)
    # sex = st.radio

    if st.button("Search"):
        if query and websites:
            task = f"Search for '{query}' of size {size} for {sex} on {', '.join(websites)}. Return a .json of the image URLs, site name, price, and other clothing properties for the top {result_count} results. Reject all cookies or keep only necessary cookies."
            # result = asyncio.run(run_agent(task))
            with st.spinner("Searching..."):
                result = asyncio.run(run_agent(task))
            st.json(result)
        else:
            st.error("Please enter a query and select at least one website.")

if __name__ == "__main__":
    main()