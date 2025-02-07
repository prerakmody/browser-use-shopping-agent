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
from typing import List
from pydantic import SecretStr, BaseModel
from urllib.parse import urlparse

# Third-party packages
import openai
import websockets
import streamlit as st
from dotenv import load_dotenv # loads from .env file containing OPENAI_API_KEY=''
from browser_use import Agent, Browser, BrowserConfig, Controller
from browser_use.browser.context import BrowserContext, BrowserContextConfig
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
KEY_OLLAMA_NA = "Ollama (local) - Not available"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_BROWSERBASE_API_KEY = "BROWSERBASE_API_KEY"
ENV_STEEL_API_KEY = "STEEL_API_KEY"
ENV_ANCHOR_API_KEY = "ANCHOR_API_KEY"

MODEL_OLLAMA_LLAMA32 = "llama3.2"
MODEL_OLLAMA_MISTRAL = "mistral"
MODEL_GPT_4O = 'gpt-4o'
MODEL_GEMINI_2_FLASH_EXP = "gemini-2.0-flash-exp"

KEY_BROWSER_CHROMIUM = "Chromium (local)"
KEY_BROWSER_BROWSERBASE = "Browserbase"
KEY_BROWSER_STEEL = "Steel.dev"
KEY_BROWSER_ANCHOR = "AnchorBrowser"

ENV_KEY_PRODUCTION = "PRODUCTION"

BOOL_PRODUCTION = True
if os.getenv(ENV_KEY_PRODUCTION) is not None:
    if os.getenv(ENV_KEY_PRODUCTION) == "False":
        BOOL_PRODUCTION = False
        print ('\n\n ===================================')
        print (" - [INFO] Running in DEBUG mode")
        print (' =================================== \n\n')

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

if not BOOL_PRODUCTION:
    install_playwright_browsers()

def run_playwright_with_xvfb():
    """
    NOTE: have not tried this yet
    """
    try:
        result = subprocess.run(["xvfb-run", "python3", "playwright_script.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error running Playwright script: {result.stderr}")
    except Exception as e:
        print(f"Exception running Playwright script: {e}")

def check_websocket_url(url):
    try:
        with websockets.connect(url) as websocket:
            websocket.send("ping")
            response = websocket.recv()
            return True
    except Exception as e:
        print(f"Error connecting to WebSocket URL: {e}")
        return False

def show_selectionui_check(verbose=False):

    if st.session_state.model_source is None or st.session_state.model_name is None or st.session_state.browser_obj is None:
        if st.session_state.model_source is None:
            if verbose:
                st.sidebar.markdown('---')
                st.sidebar.error("Please select a model source.")
        else:
            if st.session_state.model_name is None:
                if verbose: st.sidebar.error("Please select a model name.")
        if st.session_state.browser_obj is None:
            st.sidebar.markdown('---')
            if verbose: st.sidebar.error("Please select a browser.")
        return False
    else:
        return True

def extract_json(text):
    """
    Extract JSON from text
    """
    try:
        json_str = text[text.find("{"):text.rfind("}")+1]
        if len(json_str) == 0:
            return {}
        return json.loads(json_str)
    except:
        return {}

###################################################
# MODELS AND BROWSER SELECTION
###################################################

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

def set_model_source_and_name():

    try:
        
        st.session_state.model_names_ollama = get_ollama_models()
        st.session_state.model_source = st.sidebar.selectbox("Select model source:", [KEY_OPENAI, KEY_GOOGLE_GEMINI] + ([KEY_OLLAMA] if len(st.session_state.model_names_ollama) else [KEY_OLLAMA_NA]), index=None)

        if st.session_state.model_source == KEY_OPENAI:
            if os.getenv(ENV_OPENAI_API_KEY) is None:
                st.sidebar.markdown('---')
                st.sidebar.error("Please set the OPENAI_API_KEY environment variable.")
                key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
                st.sidebar.markdown('---')
                if key:
                    model_list = is_valid_openai_api_key(key)
                    if len(model_list):
                        os.environ[ENV_OPENAI_API_KEY] = key
                        st.session_state.model_names_openai = model_list
                        st.sidebar.success("OpenAI API key set successfully.")
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid OpenAI API key.")
            else:
                if len(st.session_state.model_names_openai) == 0:
                    st.session_state.model_names_openai = is_valid_openai_api_key(os.getenv(ENV_OPENAI_API_KEY))
                try:
                    model_idx = st.session_state.model_names_openai.index(MODEL_GPT_4O)
                except:
                    model_idx = None
                st.session_state.model_name = st.sidebar.selectbox("Select model:", st.session_state.model_names_openai, index=model_idx)

        elif st.session_state.model_source == KEY_GOOGLE_GEMINI:
            st.session_state.model_names_gemini = [MODEL_GEMINI_2_FLASH_EXP]
            if os.getenv(ENV_GEMINI_API_KEY) is None:
                st.sidebar.markdown('---')
                st.sidebar.error("Please set the GEMINI_API_KEY environment variable.")
                key = st.sidebar.text_input("Enter your Google Gemini API key:", type="password")
                st.sidebar.markdown('---')
                if key:
                    os.environ[ENV_GEMINI_API_KEY] = key
                    st.sidebar.success("Google Gemini API key set successfully.")
                    st.rerun()
            else:
                try:
                    model_idx = st.session_state.model_names_gemini.index(MODEL_GEMINI_2_FLASH_EXP)
                except:
                    model_idx = None
                st.session_state.model_name = st.sidebar.selectbox("Select model:", st.session_state.model_names_gemini, index=model_idx)

        elif st.session_state.model_source == KEY_OLLAMA:
            try:
                model_idx = st.session_state.model_names_ollama.index(MODEL_OLLAMA_LLAMA32)
            except:
                model_idx = None
            st.session_state.model_name = st.sidebar.selectbox("Select model:", st.session_state.model_names_ollama, index=model_idx)
        
        else:
            st.session_state.model_name = None

    except:
        traceback.print_exc()

def set_browser():

    try:
        
        st.sidebar.markdown('---')
        browser_source = st.sidebar.selectbox("Select browser source:", [KEY_BROWSER_CHROMIUM, KEY_BROWSER_BROWSERBASE, KEY_BROWSER_STEEL, KEY_BROWSER_ANCHOR], index=None)
        
        if browser_source == KEY_BROWSER_CHROMIUM:
            st.session_state.browser_obj = Browser(config=BrowserConfig())
            st.session_state.browser_choice_valid = True

        elif browser_source == KEY_BROWSER_BROWSERBASE:
            browser_obj_url = f"wss://connect.browserbase.com?apiKey={os.getenv(ENV_BROWSERBASE_API_KEY)}"
            if os.getenv(ENV_BROWSERBASE_API_KEY) is None:
                st.sidebar.error("Please set the BROWSERBASE_API_KEY environment variable.")
                key = st.sidebar.text_input("Enter your Browserbase API key:", type="password")
                if key:
                    if check_websocket_url(browser_obj_url):
                        os.environ[ENV_BROWSERBASE_API_KEY] = key
                        st.sidebar.success("Browserbase API key set successfully.")
                    else:
                        st.sidebar.error("Invalid Browserbase API key.")
                    st.rerun()
            else:
                st.session_state.browser_obj = Browser(config=BrowserConfig(
                    cdp_url=browser_obj_url
                ))

        elif browser_source == KEY_BROWSER_STEEL:
            browser_obj_url = f"wss://connect.steel.dev?apiKey={os.getenv(ENV_STEEL_API_KEY)}"
            if os.getenv(ENV_STEEL_API_KEY) is None:
                
                st.sidebar.error("Please set the STEEL_API_KEY environment variable.")
                key = st.sidebar.text_input("Enter your Steel.dev API key:", type="password")
                if key:
                    if check_websocket_url(browser_obj_url):
                        os.environ[ENV_STEEL_API_KEY] = key
                        st.sidebar.success("Steel.dev API key set successfully.")
                    else:
                        st.sidebar.error("Invalid Steel.dev API key.")
                    st.rerun()
            else:
                st.session_state.browser_obj = Browser(config=BrowserConfig(
                    cdp_url=browser_obj_url
                ))

        elif browser_source == KEY_BROWSER_ANCHOR:
            browser_obj_url = f"wss://connect.anchorbrowser.io?apiKey={os.getenv(ENV_ANCHOR_API_KEY)}"
            if os.getenv(ENV_ANCHOR_API_KEY) is None:
                st.sidebar.error("Please set the ANCHOR_API_KEY environment variable.")
                key = st.sidebar.text_input("Enter your AnchorBrowser API key:", type="password")
                if key:
                    if check_websocket_url(browser_obj_url):
                        os.environ[ENV_ANCHOR_API_KEY] = key
                        st.sidebar.success("AnchorBrowser API key set successfully.")
                    else:
                        st.sidebar.error("Invalid AnchorBrowser API key.")
                    st.rerun()
            else:
                st.session_state.browser_obj = Browser(config=BrowserConfig(
                    cdp_url=browser_obj_url
                ))

        else:
            st.session_state.browser_obj = None

    except:
        traceback.print_exc()
        pdb.set_trace()

###################################################
# CONTROLLER
###################################################

# Output controller
class Product(BaseModel):
    name: str
    price: float
    url_image: str
    url_product: str
    color: str
    material: str
    fit: str
    available_sizes: List[int]
    other_properties: str

class ProductList(BaseModel):
    site_name: str
    products: List[Product]

###################################################
# AGENTS
###################################################

def extract_messages_from_history(agent, task_placeholder):
    """
    Parameters
    ----------
    agent : Agent
        Agent object
    task_placeholder : Streamlit placeholder (st.progress(), st.empty())
    """

    try:
        
        # Step 0 - Init
        from langchain_core.messages import AIMessage

        # Step 1 - Loop ove history messages
        history_states = []
        for message in agent.message_manager.history.messages:
            if isinstance(message.message, AIMessage):
                for tool_call in message.message.tool_calls:
                    if tool_call['name'] == 'AgentOutput':
                        current_state = tool_call['args']['current_state']
                        token_count = message.metadata.input_tokens
                        current_state['token_count'] = token_count
                        history_states.append(current_state)

        # Step 2 - Display current (i.e. latest) state
        current_state = history_states[-1]
        n_steps                  = agent.n_steps
        page_summary             = current_state['page_summary']
        evaluation_previous_goal = current_state['evaluation_previous_goal']
        memory                   = current_state['memory']
        next_goal                = current_state['next_goal']
        token_count              = current_state['token_count']
        current_state_str = f"""
        - [INFO] n_steps                 : {n_steps} (token_count={token_count})
        - [INFO] page_summary            : {page_summary}
        - [INFO] evaluation_previous_goal: {evaluation_previous_goal}
        - [INFO] memory                  : {memory}
        - [INFO] next_goal               : {next_goal}
        """

        if 1:
            llm = ChatGoogleGenerativeAI(model=MODEL_GEMINI_2_FLASH_EXP, api_key=SecretStr(os.getenv(ENV_GEMINI_API_KEY)))
            response = llm.invoke(current_state_str + """\n
                            Summarize the text above so that the output looks like this. Fill the braces with information
                            **Step={}**:
                            - Previously did: {}
                            - Now doing: {}
                            """)
            current_state_str = response.content

        task_placeholder[0].progress(n_steps)
        task_placeholder[1].markdown(current_state_str)
        print ('\n ---------------')
        print (current_state_str)
        print (' --------------- \n')

    except:
        traceback.print_exc()
        pdb.set_trace()

async def run_agent(task, initial_actions, controller, task_placeholder):

    result = None

    try:

        # Step 0 - Init
        logging.info(f"\n\n ====================== [model={st.session_state.model_source, st.session_state.model_name}] Starting task: {task} \n\n")

        # Step 1 - Initialize models
        if st.session_state.model_source == KEY_OPENAI:
            llm = ChatOpenAI(model=st.session_state.model_name, timeout=90000) # model="gpt-4o-mini", "gpt-4o", "gpt-4o-turbo
        elif st.session_state.model_source == KEY_OLLAMA:
            llm = ChatOllama(model=st.session_state.model_name, num_ctx=32000)
        elif st.session_state.model_source == KEY_GOOGLE_GEMINI:
            llm = ChatGoogleGenerativeAI(model=st.session_state.model_name, api_key=SecretStr(os.getenv(ENV_GEMINI_API_KEY)))

        # Step 2 - Run agent
        agent = Agent(task=task,llm=llm
                      , max_failures=10, browser=st.session_state.browser_obj
                      , controller=controller
                      , initial_actions=initial_actions
                      ) # use_vision=True,
        
        # Step 3 - Extract results from agent
        try:
            # NOTE: relies on .run() function doing "yield self.history" (instead of return self.history) AND also a yield None in "max_steps" for loop.
            task_placeholder_all = []
            async for status in agent.run():
                extract_messages_from_history(agent, task_placeholder)
                result = status  # Capture the final result
        except:
            result = await agent.run()

        # Step 99 - Close
        logging.info(f"\n\n ====================== [model={st.session_state.model_source, st.session_state.model_name}] Completed task: {task} \n\n")

    except:
        traceback.print_exc()
        pdb.set_trace()

    return result

async def run_agents(task_list, initial_actions_list, completion_event, controller, task_placeholders):
    
    # Step 1 - Run agents
    tasks = [run_agent(task, initial_actions_list[taskId], controller, task_placeholders[taskId]) for taskId, task in enumerate(task_list)]
    
    # Step 2 - Cancel Agents (if needed)
    if st.session_state.cancel_agents:
        for task in tasks:
            task.cancel()
        print ('\n - [DEBUG] I have cancelled all the agents')
        st.write("Search cancelled.")
    
    # Step 3 - Gather output
    listHistoryAgents = await asyncio.gather(*tasks)

    # Step 4 - Set completion event
    completion_event.set()
    print ('\n - [DEBUG] I have finished all the agents')

    # Step 99 - Return results
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
        if 1:
            st.set_page_config(
                page_title="Clothing Search App",  # Tab name
                page_icon="👗",  # Optional: Tab icon
                layout="centered",  # Optional: Layout of the page ("centered" or "wide")
                initial_sidebar_state="expanded",  # Optional: Initial state of the sidebar ("expanded" or "collapsed")
            )
            st.title("Clothing Search App")

            # Step 1.2 - Streamlit sessions
            st.session_state.model_source = None
            st.session_state.model_name = None
            st.session_state.model_names_openai = []
            st.session_state.model_names_gemini = []
            st.session_state.model_names_ollama = []
            st.session_state.browser_obj = None
            if 'cancel_agents' not in st.session_state:
                st.session_state.cancel_agents = False

        ## ---------------- Step 1.2 - Streamlit UI (get model and browser)
        if 1:
            
            set_model_source_and_name()
            set_browser()
            
        ## ---------------- Step 1.99 - Streamlit UI (User input)
        if show_selectionui_check(verbose=True):
            query = st.text_input("Enter your query:", help="e.g. 'black t-shirt, gray pantaloons, jumpsuit, troyer collar etc'")
            size = st.selectbox("Select size:", ["S", "M", "L"])
            sex = st.radio("Select sex:", ["Male", "Female", "Unisex"])
            websites = st.multiselect("Select websites to query:", ["https://www.zalando.nl", "https://www.hm.com", "https://www.zara.nl"])
            result_count = st.slider("Number of results:", 1, 15, 1)

    ## ---------------- Step 2 - Streamlit UI (Search button)
    if show_selectionui_check(verbose=False):
        
        if st.button("Cancel"):
            st.session_state.cancel_agents = True

        if st.button("Search", use_container_width=True):

            st.session_state.cancel_agents = False

            print (f'\n\n ================================= \n\n')
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
                    task_list = [
                        # f"""Search for '{query}' of size {size} for {sex} on {website}.
                        f"""
                        MAIN TASK:
                        Open {website}, reject all cookies (if prompted) and convert language to English. If you cannot convert language to English, then just proceed.
                        Then find the search bar, input the search term = '{query}' and hit Enter. if the search bar cannot be found, move on to the next step.
                        Then filter with size={size} and sex={sex}.

                        FURTHER INSTRUCTIONS:
                        Look at the top {result_count} results and visit the webpage for each item.
                        Parse these items' webpages and ensure that it matches the search term = '{query}'. 
                        Then return a .json with the primary keys as 'site_name' and 'products'. 
                        The 'products' key is a list of items with the following keys:
                        'name', 'url_product', 'url_image', 'price', 'color', 'material', 'available_sizes', 'fit' and 'other_properties'. 
                        Refine the results for color, material, available sizes, and other properties by opening each products page.
                        Ignore delivery, return, care instructions, reviews, and other irrelevant information.
                        
                        OTHER INSTRUCTONS:
                        Also, dont scroll too much to filter with size={size} and sex={sex}. These are usually present on the top (or left-side) of the page
                        """
                        for website in websites
                    ]
                    st.sidebar.markdown('---')
                    st.sidebar.text("Task Prompt:")
                    # st.sidebar.write(task_list[0])
                    st.sidebar.markdown(task_list[0])

                    initial_actions_list = [ [{'open_tab': {'url': website}}] for website in websites ]
                    
                    task_placeholders = []
                    for _ in task_list:
                        st.markdown('---')
                        task_placeholders.append((st.progress(0), st.empty()))
                    # task_placeholders = [st.empty() for _ in task_list]

                    controller = Controller(output_model=ProductList)

                ## ---------------- Step 5.1 - Run agents
                if len(debugOutput) == 0:
                    log_placeholder = st.empty()
                    completion_event = asyncio.Event()
                    tStart = time.time()
                    with st.spinner("Searching..."):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        results = loop.run_until_complete(
                            asyncio.gather(run_agents(task_list, initial_actions_list, completion_event, controller, task_placeholders), update_logs(log_placeholder, completion_event))
                        )
                        listHistoryAgents = results[0]  # Get the results from the first coroutine
                        print ('\n\n ========== \n len(listHistoryAgents): ', len(listHistoryAgents), '\n\n ==========\n\n')
                    timeTaken = time.time() - tStart
                    st.write(f"Time taken: {timeTaken:.2f} seconds")
                    st.markdown('---')

                ## ---------------- Step 5.2 - Combine agent results
                combined_results = {}
                if len(debugOutput) == 0:
                    
                    print ('\n\n ================ Parsing results ================= \n\n')
                    for historyAgentObj in listHistoryAgents: # type(result) == browser_use.agent.views.AgentHistoryList
                        if historyAgentObj is None:
                            print (' - [INFO] historyAgentObj is None. Skipping')
                            continue

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
                                else:
                                    # sometimes the JSON is there but just not with is_done=True
                                    stepContentJSON = extract_json(actionResult.extracted_content)
                                    pprint.pprint(stepContentJSON)
                                
                            except:
                                pass
                        print ('\n\n ================================= \n\n')
                else:
                    for debugJson in debugOutput:
                        combined_results[debugJson['site_name']] = debugJson['products']

                # Step 5.2.99 - Debug
                st.sidebar.markdown('---')
                st.sidebar.text("Output json:")
                st.sidebar.json(combined_results, expanded=False)
                
                ## ---------------- Step 6 - Display results
                # if not BOOL_PRODUCTION: pdb.set_trace() # check results in debug mode
                if 1:

                    num_columns = 3  # Number of columns in the grid
                    columns = st.columns(num_columns)

                    try:
                        for siteName in combined_results:
                            st.subheader(siteName)
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
                                            if available_sizes is not None:
                                                if len(available_sizes) > 0:
                                                    available_sizes = [str(each) for each in available_sizes]
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