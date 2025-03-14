import google.generativeai as genai
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.getipython import get_ipython
import sys
from io import StringIO
import time
import prophet
import os 
import imblearn
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("api_key"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
chat_instructions = model.start_chat(history=[])
chat_code = model.start_chat(history=[])
InteractiveShell.ast_node_interactivity = "all"

df = pd.read_csv("dataset.csv")

dataset = "dataset.csv"
algortihm = ""
user_prompt = """

The data provided by the company states that the company uses a pricing model that only takes the expected ride duration as a factor to determine the price for a ride. Now, we will implement a dynamic pricing strategy aiming to adjust the ride costs dynamically based on the demand and supply levels observed in the data. It will capture high-demand periods and low-supply scenarios to increase prices, while low-demand periods and high-supply situations will lead to price reductions.
"""


system1 = f"""
Assume you are guiding someone through the process of training a machine learning model. Provide step-by-step instructions for each phase of the journey.

*Dataset is '{dataset}'*
"""+"""
- If algorithm is provided, then use that algorithm directly.
- Deliver one instruction at a time in separate prompts.
- Avoid providing Python code; focus on providing clear, actionable directions.
- If there are any previous results or code executions, incorporate those into the instructions where relevant.
- Start by importing libraries, loading the dataset, and performing initial data exploration.
- Dont do visualization, instead of that create a tables.
- If the prompt contains an error, summarize the problem and instruct how to fix it.
- If Error Occurs, then repeat the last prompt in a simpler way.
- If Error Occurs in last 5 prompts, then reply 'Q2xlYXIgbXkgbWVtb3J5', so that i can start over.
- At the end, take accuracy and give a json {'algorithm':'alg_name', 'accuracy':'accuracy_percentage'}
- Avoid Printing big Dataframes and lists or big results. instead print that details of head or tail.
- After choosing Model, export the last dataframe to a csv file 'latest_dataset.csv', Give the entire remaining code in the next prompt.
- Dont use Notebook magic commands (e.g., %matplotlib inline).
- Dont use Notebook only functions (e.g., display()).
Remember: While instructing, dont tell to assume anything, instead give the exact instruction after studying resutls, or else ask if you need anything else.
Remember: dont give column names while instructing, instead give the column index number.
"""

system2 = """
Imagine you are an ML Engineer working in a Python notebook. Your task is to analyze the data step-by-step.

Please provide each piece of Python code sequentially, focusing on a single cell of analysis per response. 
- Each response should be a complete, executable cell and should not include long code spans or multiple cells of code.

Remember: Dont ask wwhy, Do it blindly, I will tell you why and whats the output sequentially, Give me code i want and shut.

- Dont do visualization, instead of that create a tables.
- Dont use Notebook magic commands (e.g., %matplotlib inline).
- Dont use Notebook only functions (e.g., display()).
- Dont make any Syntax errors.
"""


import signal

def execute_code(code_string, timeout=60):
    def handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    try:
        code = code_string.strip()
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        exec(code, globals())
        signal.alarm(0)
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        if '=' not in code and output == "":
            result = eval(code, globals())
            return str(result)
        if output:
            return output.strip()

        return "Code executed successfully"

    except TimeoutError as e:
        sys.stdout = old_stdout
        return f"Error: {str(e)}"
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error Executing: {str(e)}"

with open("chathistory.txt", "w") as chathistory:
    initial_prompt = f"""Follow the steps below to perform the analysis and build the prediction model.

    {"" if algortihm== "" else f'use {algortihm} algorithm to predict the monthly trends of the retail shop.'} 
    
    Step 1: Data Preprocessing and Exploration
    Inspect the dataset and check for any missing values or data inconsistencies.
    Perform data reduction, removing irrelevant features if necessary.
    Identify potential patterns in the data that could be used to predict the monthly trends of the retail shop.

    {user_prompt}

    Last Step: Get Accuracy

    Please begin with Step 1: inspecting and exploring the data.
    """

    current_prompt = initial_prompt
    step = 1
    last_code_output = ""

    while True:
        if last_code_output:
            full_prompt = f"{current_prompt}\nPrevious code execution result: {last_code_output}"
        else:
            full_prompt = current_prompt
        time.sleep(3)

        response1 = chat_instructions.send_message(f"<system>{system1}</system> {full_prompt}")
        print(f"Step {step} Instruction:\n{response1.text}")
        if 'Q2xlYXIgbXkgbWVtb3J5' in response1.text:
            exit()
        elif 'json' in response1.text:
            exit()
        chathistory.write(f"Step {step} Instruction:\n{response1.text}\n\n")
        time.sleep(3)
        
        response2 = chat_code.send_message(f"<system>{system2}</system> {response1.text}")
        print(f"Step {step} Code:\n{response2.text}")
        time.sleep(3)

        code_snip = response2.text.replace("```python", "").replace("```", "")
        code_output = execute_code(code_snip)
        print(f"Step {step} Output:\n{code_output}")
        time.sleep(3)

        chathistory.write(f"Step {step} Code:\n{code_snip}\n")
        chathistory.write(f"Step {step} Output:\n{code_output}\n\n\nNext Step\n\n")

        last_code_output = code_output

        current_prompt = "continue"
        step += 1