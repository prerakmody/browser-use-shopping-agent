import asyncio

from dotenv import load_dotenv
from browser_use import Agent
from langchain_openai import ChatOpenAI

load_dotenv()

async def main():
    agent = Agent(
        # task="Go to zalando.nl, search for 'troyer collar', and download and save the images of the top 5 options of size S.",
        # task="Go to zalando.nl, search for waterproof shoes of size 42. Returns a .json of the image URLs and product description.",
        task="List the document requirements for Overseas Citizen of India (OCI) card. Search this in google.com and summarize the results. Also, return the references",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    print(result)
    import pdb; pdb.set_trace() 
    # result.action_results()[-1].extracted_content

asyncio.run(main())

"""
pip install markdownify
"""