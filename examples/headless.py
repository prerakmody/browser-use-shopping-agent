"""
This example demonstrates how to use the Browser class to interact with a headless browser.
Here we use the "https://www.browserbase.com/overview"
"""

# Standard library
import os
import pdb
import asyncio
from typing import List
from pydantic import SecretStr, BaseModel

# 3rd party packages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig, Controller
load_dotenv()

# Constants
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_BROWSERBASE_API_KEY = "BROWSERBASE_API_KEY"
ENV_STEEL_API_KEY = "STEEL_API_KEY"

# Create a browser instance
browser = Browser(
    config=BrowserConfig(
        cdp_url=f"wss://connect.browserbase.com?apiKey={os.getenv(ENV_BROWSERBASE_API_KEY)}"
        # cdp_url=f"wss://connect.steel.dev?apiKey={os.getenv(ENV_STEEL_API_KEY)}"
    )
)

# Output controller
class Item(BaseModel):
	item_name: str
	item_url: str
	item_image_urls: List[str]

async def main():

    # Step 1 - Pick a model
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=SecretStr(os.getenv(ENV_GEMINI_API_KEY)))
    llm=ChatOpenAI(model="gpt-4o")

    # Step 2 - Set the controller
    controller = Controller(output_model=Item)

    # Step 2 - Create an agent
    agent = Agent(
        task="""
        Go to zalando.nl, reject cookies and change language to English if possible. 
        Then, find the search bar, input 'troyer collar' and hit Enter. 
        Click on the first item in the result and return the image url.
        """
        , llm=llm, max_failures=10
        , browser=browser
        , controller=controller
    )

    # Step 3 - Run the agent
    history = await agent.run()

    # Step 4 - Process the results
    print(history) # {..final_result(), .action_results[0].result, .action_results[1].extracted_content}
    pdb.set_trace()

if __name__ == "__main__":
    asyncio.run(main())