# browser-use-shopping-agent
Send the same query to multiple shopping websites and displays the results

# Installation
1. Get OpenAI API key [here](https://platform.openai.com/settings/organization/usage). Or a gemini key [here](https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/metrics)
2. Install packages
```bash
pip instal browser_use, streamlit, dotenv
pip install openai

playwright install # installs browsers
```
3. Make a .env file with (any of) the following keys
```bash
OPENAI_API_KEY="sk-..."
GEMINI_API_KEY="..."
BROWSERBASE_API_KEY="..."
STEEL_API_KEY="..."
ANCHOR_API_KEY="..."
```

# Models
1. Get OpenAI keys [here](https://platform.openai.com/api-keys)
2. Get Gemini keys [here](https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/metrics)
3. Get ollama [here](https://ollama.com/search?c=tools)
    - `ollama pull llama3.2`
    - `ollama pull mistral`
      ```python
        from langchain_ollama import ChatOllama
        llm=ChatOllama(model="llama3.2", num_ctx=32000) # llama3.2, mistral
        messages = [ ( "system", "You are a helpful assistant that translates English to French. Translate the user sentence.", ), ("human", "I love programming."), ]
        llm.invoke(messages)
      ```

# Usage
```bash
streamlit run main.py
# OR
streamlit run main.py --server.address 0.0.0.0
```

<video src="https://github.com/user-attachments/assets/9d457eee-0ab0-4c85-8e04-51b3da8790eb" width="352" height="720"></video>





# Future Stuff
1. [P] https://github.com/emmetify/emmetify-py
2. [P] Track the steps of the agent, specifically the "Eval" and "Next Goal" parts
