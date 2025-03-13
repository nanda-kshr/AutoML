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
import signal
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("api_key"))
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
chat_instructions = model.start_chat(history=[])
chat_code = model.start_chat(history=[])
InteractiveShell.ast_node_interactivity = "all"

df = pd.read_csv("dataset.csv")

system1 = """
Assume you are an expert ML practitioner guiding someone through training a high-performance machine learning model. Provide clear instructions.

*Dataset is 'dataset.csv'*

- Deliver one instruction at a time that accomplishes a meaningful step.
- Avoid providing Python code; focus on providing clear, actionable directions.
- Maintain context awareness - reference previous steps and variables that have been created.
- If there are any previous results or code executions, incorporate them into planning the next step.
- Start with essential libraries and gradually build complexity.
- Don't do visualization, instead create concise tables with key statistics.
- If the prompt contains an error, diagnose the issue and suggest specific fixes.
- If Error Occurs, address the specific error rather than simplifying too much.
- If Error Occurs in last 5 prompts, then reply 'Q2xlYXIgbXkgbWVtb3J5', so that i can start over.
- Break complex ML processes into clear logical steps (preprocessing, feature engineering, model training, evaluation).
- At the end, calculate accuracy and give a json {'algorithm':'alg_name', 'accuracy':'accuracy_percentage'}
- Print only small samples of data (df.head(5)) and concise summaries.
- After choosing Model, export the dataframe to 'latest_dataset.csv'.
- Don't use Notebook magic commands (e.g., %matplotlib inline).
- Don't use Notebook only functions (e.g., display()).
"""

system2 = """
You are an ML Engineer implementing a complete analysis. Your task is to write proper, executable code segments.

IMPORTANT GUIDELINES:
- Each code snippet should be 5-15 lines - complete enough to accomplish one logical step
- Maintain state and memory of previous variables and operations
- When referencing variables, ONLY use variables that were previously defined
- Include basic error handling for common issues
- Focus on one clear task per code segment (e.g., data loading, cleaning, feature creation)
- Balance between brevity and completeness - each snippet should do something meaningful
- Always check if required variables exist before using them
- Print small data samples only (first 5 rows, not entire dataframes)
- Include brief inline comments only for critical operations
- Don't use visualization code
- Each code segment should be completely executable on its own
- Reference previous results when building upon them
- Don't use Notebook magic commands or display() functions
- Always ensure variable consistency between code segments
"""


def execute_code(code_string, timeout=30):
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
    initial_prompt = """Follow the steps below to perform the analysis and build the prediction model.

    Step 1: Data Preprocessing and Exploration
    Inspect the dataset and check for any missing values or data inconsistencies.
    Perform data reduction, removing irrelevant features if necessary.
    Identify potential patterns in the data that could be used to predict the monthly trends of the retail shop.

    Step 2: Model Training
    Train a machine learning model (such as a time series model or regression model) on the transaction data to predict monthly trends.
    Choose an appropriate model based on the data characteristics and goals.

    Step 3: Model Evaluation
    Evaluate the trained model by comparing the actual transaction data and the predicted values.
    Visualize both the actual and predicted data in a clear, easily interpretable manner (e.g., using a line chart or bar graph).

    Step 4: Prediction of Monthly Trends
    Use the trained model to predict future monthly trends for the retail shop.
    Provide insights on the predicted trends and their potential impact on business strategies.
    Ensure to follow each step in a logical sequence and provide Python code for each phase.

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