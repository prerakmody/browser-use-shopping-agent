# -----------------------------------------------------------------------------
# (Multi-website) Clothing Search App
# Author: Prerak Mody (https://github.com/prerakmody/)
# License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# -----------------------------------------------------------------------------

# Standard library
import os
import pdb
import time
import json
import pprint
import asyncio
import logging
import subprocess
import traceback
from pathlib import Path
from io import StringIO
from pydantic import SecretStr
from urllib.parse import urlparse

# Third-party packages
import openai
import streamlit as st
from dotenv import load_dotenv # loads from .env file containing OPENAI_API_KEY=''
from browser_use import Agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# Step 0.1 - Init (load env variables)
load_dotenv()

# Step 0.2 - Init (Set up logging) # TODO: Currently not streaming logs to the streamlit UI
log_stream = StringIO()
logging.basicConfig(stream=log_stream, level=logging.INFO)

# Step 0.3 - Constants
DIR_THIS = Path(__file__).resolve().parent
DIR_ASSETS = DIR_THIS / "assets"
DEBUG_JSON_PATHS = [] # [], ['sample-zalando.json']
debugOutput = []
if len(DEBUG_JSON_PATHS) > 0:
    for debugJsonPath in DEBUG_JSON_PATHS:
        with open(DIR_ASSETS / debugJsonPath, 'r') as f:
            debugOutput.append(json.load(f))

KEY_OPENAI = "Open AI"
KEY_GOOGLE_GEMINI = "Google Gemini"
KEY_OLLAMA = "Ollama (local)"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"

OLLAMA_MODEL_LLAMA32 = "llama3.2"
OLLAMA_MODEL_MISTRAL = "mistral"

###################################################
# UTILS
###################################################

def install_playwright_browsers():
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
        subprocess.run(["playwright", "install", "firefox"], check=True)
        subprocess.run(["playwright", "install", "webkit"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing Playwright browsers: {e}")

install_playwright_browsers()

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_valid_openai_api_key(api_key):
    
    model_list = []
    try:
        openai.api_key = api_key
        model_list = openai.models.list() # Make a simple request to check the validity of the API key (https://platform.openai.com/settings/organization/limits)
        model_list = sorted([each.id for each in model_list.data])
    except:
        return []
    
    return model_list

def get_models_openai():

    model_list = []    
    try:
        model_list = sorted(st.session_state.model_list_openai)
        
    except:
        return is_valid_openai_api_key(os.getenv('OPENAI_API_KEY'))

    return model_list

def get_ollama_models():
    """
    NAME               ID              SIZE      MODIFIED
    mistral:latest     f974a74358d6    4.1 GB    3 minutes ago
    llama3.2:latest    a80c4f17acd5    2.0 GB    28 minutes ago
    """

    ollama_models = []

    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            ollama_list = result.stdout.splitlines()
            ollama_models = [each.split(' ')[0].split(':latest')[0] for each in ollama_list[1:]]
    
    except Exception as e:
        print(f"Error running 'ollama list': {e}")
        return []

    return ollama_models

###################################################
# AGENTS
###################################################

async def run_agent(task, model_source, modelname):

    result = None

    try:
        logging.info(f"\n\n ====================== [model={model_source, modelname}] Starting task: {task} \n\n")
        if model_source == KEY_OPENAI:
            llm = ChatOpenAI(model=modelname, timeout=90000) # model="gpt-4o-mini", "gpt-4o", "gpt-4o-turbo
        elif model_source == KEY_OLLAMA:
            llm = ChatOllama(model=modelname, num_ctx=32000)
        elif model_source == KEY_GOOGLE_GEMINI:
            llm = ChatGoogleGenerativeAI(model=modelname, api_key=SecretStr(os.getenv(ENV_GEMINI_API_KEY)))

        agent = Agent(task=task,llm=llm, max_failures=10) # use_vision=True,
        result = await agent.run()

        logging.info(f"\n\n ====================== [model={model_source, modelname}] Completed task: {task} \n\n")

    except:
        traceback.print_exc()

    return result

async def run_agents(tasks, completion_event, model_source, modelname):
    listHistoryAgents = await asyncio.gather(*[run_agent(task, model_source, modelname) for task in tasks])
    completion_event.set()
    print ('\n - [DEBUG] I have finished all the agents')
    return listHistoryAgents

async def update_logs(log_placeholder, completion_event):
    logging.info("Starting log update loop")
    while not completion_event.is_set():
        # log_placeholder.text(log_stream.getvalue())
        log_placeholder.write(log_stream.getvalue())
        await asyncio.sleep(0.5)  # Update logs every 0.5 seconds

###################################################
# MAIN
###################################################

def main():

    ## ---------------- Step 1 - Streamlit UI (Sidebar)
    if 1:
        ## ---------------- Step 1.1 - Streamlit UI (title)
        st.title("Clothing Search App")

        ## ---------------- Step 1.2 - Streamlit UI (get model)
        models_ollama = get_ollama_models()
        model_source = st.sidebar.selectbox("Select model source:", [KEY_OPENAI, KEY_GOOGLE_GEMINI] + ([KEY_OLLAMA] if len(models_ollama) else []))
        st.session_state.model_source_valid = False

        if model_source == KEY_OPENAI:
            if os.getenv(ENV_OPENAI_API_KEY) is None:
                st.sidebar.markdown('---')
                st.sidebar.error("Please set the OPENAI_API_KEY environment variable.")
                key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
                st.sidebar.markdown('---')
                if key:
                    model_list = is_valid_openai_api_key(key)
                    if len(model_list):
                        os.environ[ENV_OPENAI_API_KEY] = key
                        # set model_list in streamlit cache
                        st.session_state.model_list_openai = model_list
                        st.session_state.model_source_valid = True
                        st.sidebar.success("OpenAI API key set successfully.")
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid OpenAI API key.")
            else:
                st.session_state.model_list_openai = get_models_openai()
                st.session_state.model_source_valid = True

        elif model_source == KEY_GOOGLE_GEMINI:
            st.session_state.model_list_openai = ["gemini-2.0-flash-exp"]
            if os.getenv(ENV_GEMINI_API_KEY) is None:
                st.sidebar.markdown('---')
                st.sidebar.error("Please set the GEMINI_API_KEY environment variable.")
                key = st.sidebar.text_input("Enter your Google Gemini API key:", type="password")
                st.sidebar.markdown('---')
                if key:
                    os.environ[ENV_GEMINI_API_KEY] = key
                    st.session_state.model_source_valid = True
                    st.sidebar.success("Google Gemini API key set successfully.")
                    st.rerun()
            else:
                st.session_state.model_source_valid = True

        elif model_source == KEY_OLLAMA:
            st.session_state.model_list_ollama = models_ollama
            st.session_state.model_source_valid = True

        ## ---------------- Step 1.3 - Streamlit UI (User input)
        if st.session_state.model_source_valid:
            
            query = st.text_input("Enter your query:", help="e.g. 'black t-shirt, gray pantaloons, jumpsuit, troyer collar etc'")
            size = st.selectbox("Select size:", ["S", "M", "L"])
            sex = st.radio("Select sex:", ["Male", "Female", "Unisex"])
            websites = st.multiselect("Select websites to query:", ["zalando.nl", "hm.com", "zara.nl"])
            result_count = st.slider("Number of results:", 5, 15, 5)
            if model_source == KEY_OPENAI:
                st.session_state.model_list_openai = get_models_openai()
                model_idx = st.session_state.model_list_openai.index("gpt-4o-mini")
                modelname = st.selectbox("Select model:", st.session_state.model_list_openai, index=model_idx)
            elif model_source == KEY_GOOGLE_GEMINI:
                model_idx = st.session_state.model_list_openai.index("gemini-2.0-flash-exp")
                modelname = st.selectbox("Select model:", st.session_state.model_list_openai, index=model_idx)
            elif model_source == KEY_OLLAMA:
                model_idx = st.session_state.model_list_ollama.index(OLLAMA_MODEL_LLAMA32)
                modelname = st.selectbox("Select model:", st.session_state.model_list_ollama, index=model_idx)

    ## ---------------- Step 2 - Streamlit UI (Search button)
    if st.session_state.model_source_valid:
    
        if st.button("Search"):

            print (f'\n\n ================ modelname={modelname} ================= \n\n')
            print (' - [INFO] query:', query)
            print (' - [INFO] size:', size)
            print (' - [INFO] sex: ', sex)
            print (' - [INFO] websites:', websites)
            print (' - [INFO] result_count:', result_count)
            print ('\n\n ================================= \n\n')

            ## ---------------- Step 3 - Streamlit UI (Search results)
            if (query and websites) or len(debugOutput):

                ## ---------------- Step 4 - Setup prompt (TODO: need more prompt finetuning)
                if len(debugOutput) == 0:
                    tasks = [
                        # f"""Search for '{query}' of size {size} for {sex} on {website}.
                        f"""Open {website}, reject all cookies (if prompted) and convert language to English. If you cannot convert language to English, then just proceed.
                        Then find the search bar and input the search term = '{query}'. 
                        Then filter with size={size} and sex={sex}.
                        Look at the top {result_count} results and visit the webpage for each item.
                        Parse these items' webpages and return a .json with the primary keys as 'site_name' and 'products'. 
                        The 'products' key is a list of items with the following keys:
                        'name', 'url_product', 'url_image', 'price', 'color', 'material', 'available_sizes', 'fit' and 'other_properties'. 
                        Refine the results for color, material, available sizes, and other properties by opening each products page.
                        Ignore delivery, return, care instructions, reviews, and other irrelevant information.
                        """
                        for website in websites
                    ]

                ## ---------------- Step 5.1 - Run agents
                if len(debugOutput) == 0:
                    log_placeholder = st.empty()
                    completion_event = asyncio.Event()
                    tStart = time.time()
                    with st.spinner("Searching..."):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(
                            asyncio.gather(run_agents(tasks, completion_event, model_source, modelname), update_logs(log_placeholder, completion_event))
                        )
                        listHistoryAgents = results[0]  # Get the results from the first coroutine
                    timeTaken = time.time() - tStart
                    st.write(f"Time taken: {timeTaken:.2f} seconds")

                ## ---------------- Step 5.2 - Combine agent results
                combined_results = {}
                if len(debugOutput) == 0:
                    
                    print ('\n\n ================ Parsing results ================= \n\n')
                    for historyAgentObj in listHistoryAgents: # type(result) == browser_use.agent.views.AgentHistoryList
                        resultObjList = historyAgentObj.action_results()
                        print (f'\n\n ================ resultObj (len={len(resultObjList)}) ================= \n\n')
                        for stepId, actionResult in enumerate(resultObjList):
                            try:
                                print (f' - [step={stepId+1}] done: {actionResult.is_done}')
                                if actionResult.is_done:
                                    print (f'   -- [step={stepId+1}] done: {actionResult.is_done}')
                                    actionJson = json.loads(actionResult.extracted_content)
                                    siteName = actionJson["site_name"]
                                    print (f'\n\n ---------- {siteName} ---------- \n\n')
                                    pprint.pprint(actionJson)
                                    combined_results[siteName] = actionJson["products"]
                                
                            except:
                                pass
                        print ('\n\n ================================= \n\n')
                else:
                    for debugJson in debugOutput:
                        combined_results[debugJson['site_name']] = debugJson['products']

                # Step 5.2.99 - Debug
                st.json(combined_results, expanded=False)
                
                ## ---------------- Step 6 - Display results
                pdb.set_trace()
                if 1:

                    num_columns = 3  # Number of columns in the grid
                    columns = st.columns(num_columns)

                    try:
                        for siteName in combined_results:
                            for idx, item in enumerate(combined_results[siteName]):
                                try:
                                    col = columns[idx % num_columns]
                                    with col:
                                        if is_valid_url(item["url_image"]):
                                            st.image(item["url_image"], caption=f"{item['name']} - {item['price']}")
                                            color = item.get("color", "transparent")  # Default to transparent if color is not available
                                            st.markdown(f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; display: inline-block;"></div>', unsafe_allow_html=True)
                                            st.markdown(f"[View Product]({item['url_product']})")
                                            st.markdown(f"**Material:** {item.get('material', 'N/A')}")
                                            available_sizes = item['available_sizes'] if item.get('available_sizes') is not None else []
                                            st.markdown(f"**Available Sizes:** {', '.join(available_sizes)}")
                                        else:
                                            st.write(f"Image not available for {item['name']}")
                                except:
                                    traceback.print_exc()
                                    pdb.set_trace()
                    except:
                        traceback.print_exc()

                    # pdb.set_trace()

            else:
                st.error("Please enter a query and select at least one website.")
    
    else:
        pass

if __name__ == "__main__":
    main()

"""
To run
 - streamlit run clothing-multibrowser.py
 - ollama pull llama3.2 : http://localhost:11434/
 - ollama list
"""
"""
>> python
from langchain_ollama import ChatOllama
llm=ChatOllama(model="llama3.2", num_ctx=32000) # llama3.2, mistral
messages = [ ( "system", "You are a helpful assistant that translates English to French. Translate the user sentence.", ), ("human", "I love programming."), ]
llm.invoke(messages)
"""
"""
To-check
- https://github.com/emmetify/emmetify-py
- https://github.com/browser-use/browser-use/blob/0.1.34/docs/customize/output-format.mdx
- initial_actions
"""