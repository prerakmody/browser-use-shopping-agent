# Standard library
import os
import pdb
import time
import json
import pprint
import asyncio
import logging
import traceback
from pathlib import Path
from io import StringIO

# Third-party packages
import streamlit as st
from dotenv import load_dotenv # loads from .env file containing OPENAI_API_KEY=''
from browser_use import Agent
from langchain_openai import ChatOpenAI

# Step 0.1 - Init (load env variables)
load_dotenv()

# Step 0.2 - Init (Set up logging) # TODO: Currently not streaming logs to the streamlit UI
log_stream = StringIO()
logging.basicConfig(stream=log_stream, level=logging.INFO)

# Step 0.3 - Constants
DIR_THIS = Path(__file__).resolve().parent
DIR_ASSETS = DIR_THIS / "assets"
DEBUG_JSON = None # 'sample-zalando.json'
debugJson = None
if len(DEBUG_JSON) > 0:
    with open(DIR_ASSETS / DEBUG_JSON, 'r') as f:
        debugJson = json.load(f)

###################################################
# AGENTS
###################################################
async def run_agent(task):
    logging.info(f"\n\n ====================== Starting task: {task} \n\n")

    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o", timeout=90000),
    )
    result = await agent.run()
    logging.info(f"\n\n ======================  Completed task: {task} \n\n")
    return result

async def run_agents(tasks, completion_event):
    results = await asyncio.gather(*[run_agent(task) for task in tasks])
    completion_event.set()
    print ('\n - [DEBUG] I have finished all the agents')
    return results

async def update_logs(log_placeholder, completion_event):
    while not completion_event.is_set():
        log_placeholder.text(log_stream.getvalue())
        await asyncio.sleep(0.5)  # Update logs every 0.5 seconds

###################################################
# MAIN
###################################################

def main():

    ## ---------------- Step 1 - Streamlit UI (Sidebar)
    if 1:
        ## ---------------- Step 1.1 - Streamlit UI (title)
        st.title("Clothing Search App")

        ## ---------------- Step 1.2 - Streamlit UI (User input)
        query = st.text_input("Enter your query:")
        size = st.selectbox("Select size:", ["S", "M", "L"])
        sex = st.radio("Select sex:", ["Male", "Female", "Unisex"])
        websites = st.multiselect("Select websites to query:", ["zalando.nl", "hm.com", "zara.nl"])
        result_count = st.slider("Number of results:", 5, 15, 5)
        if os.getenv('OPENAI_API_KEY') is None:
            st.error("Please set the OPENAI_API_KEY environment variable.")
            key = st.text_input("Enter your OpenAI API key:", type="password")
            os.environ["OPENAI_API_KEY"] = key
            return

    ## ---------------- Step 2 - Streamlit UI (Search button)
    if st.button("Search"):

        ## ---------------- Step 3 - Streamlit UI (Search results)
        if (query and websites) or debugJson is not None:

            ## ---------------- Step 4 - Setup prompt (TODO: need more prompt finetuning)
            if debugJson is None:
                tasks = [
                    f"""Search for '{query}' of size {size} for {sex} on {website}. 
                    Return a .json with the primary keys as 'site_name' and 'products'. In products is a list of items with the following keys:
                    'name', 'url_product', 'url_image', 'price', 'color', 'material', 'available_sizes', 'other_properties'.
                    Do this for the top {result_count} results. 
                    Reject all cookies or keep only necessary cookies."""
                    for website in websites
                ]

            ## ---------------- Step 5.1 - Run agents
            if debugJson is None:
                # log_placeholder = st.empty()
                # with st.spinner("Searching..."):
                #     resultsList = asyncio.run(run_agents(tasks))
                log_placeholder = st.empty()
                completion_event = asyncio.Event()
                tStart = time.time()
                with st.spinner("Searching..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    resultsList = loop.run_until_complete(
                        asyncio.gather(run_agents(tasks, completion_event), update_logs(log_placeholder, completion_event))
                    )
                    resultsList = resultsList[0]  # Get the results from the first coroutine
                timeTaken = time.time() - tStart
                st.write(f"Time taken: {timeTaken:.2f} seconds")

            ## ---------------- Step 5.2 - Combine agent results
            combined_results = {}
            if debugJson is None:
                
                print ('\n\n ================ Parsing results ================= \n\n')
                for resultObj in resultsList: # type(result) == browser_use.agent.views.AgentHistoryList
                    print ('\n\n ================================= \n\n')
                    for actionResult in resultObj.action_results():
                        try:
                            actionJson = json.loads(actionResult.extracted_content)
                            print ('\n\n -------------------- \n\n')
                            pprint.pprint(actionJson)
                            siteName = actionJson["site_name"]
                            combined_results[siteName] = actionJson["products"]
                            
                        except:
                            pass
                    print ('\n\n ================================= \n\n')
            else:
                combined_results[debugJson['site_name']] = debugJson['products']

            # Step 5.2.99 - Debug
            st.json(combined_results, expanded=False)
            
            ## ---------------- Step 6 - Display results
            pdb.set_trace()
            if 1:
                # for item in combined_results["results"]:
                #     st.image(item["url_image"], caption=f"{item['name']} - {item['price']} - {item['site_name']} - {item['size']} - {item['color']} - {item['material']}")

                # for siteName in combined_results:
                #     for item in combined_results[siteName]:
                #         try:
                #             st.image(item["url_image"], caption=f"{item['name']} - {item['price']}")
                #             color = item.get("color", "transparent")  # Default to transparent if color is not available
                #             st.markdown(f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>', unsafe_allow_html=True)
                #             st.markdown(f"[View Product]({item['url_product']})")
                #             st.markdown(f"**Material:** {item.get('material', 'N/A')}")
                #             st.markdown(f"**Available Sizes:** {', '.join(item.get('available_sizes', []))}")
                #         except:
                #             traceback.print_exc()
                #             pdb.set_trace()

                num_columns = 3  # Number of columns in the grid
                columns = st.columns(num_columns)

                for siteName in combined_results:
                    for idx, item in enumerate(combined_results[siteName]):
                        try:
                            col = columns[idx % num_columns]
                            with col:
                                st.image(item["url_image"], caption=f"{item['name']} - {item['price']}")
                                color = item.get("color", "transparent")  # Default to transparent if color is not available
                                st.markdown(f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>', unsafe_allow_html=True)
                                st.markdown(f"[View Product]({item['url_product']})")
                                st.markdown(f"**Material:** {item.get('material', 'N/A')}")
                                st.markdown(f"**Available Sizes:** {', '.join(item.get('available_sizes', []))}")
                        except:
                            traceback.print_exc()
                            pdb.set_trace()
        
        else:
            st.error("Please enter a query and select at least one website.")

if __name__ == "__main__":
    main()

"""
Ti run
 - streamlit run clothing-multibrowser.py
"""