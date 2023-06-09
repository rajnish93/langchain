# from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv 
import json
import streamlit as st
# load_dotenv()


from torch import cuda, bfloat16
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM 
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'  
# device =  'cpu'  
print("device",device)
#creating a model 
fmodel = AutoModelForCausalLM.from_pretrained(
    'tiiuae/falcon-7b-instruct',
    trust_remote_code=True,
    torch_dtype=float

)
fmodel.eval() 
fmodel.to(device) 
print(f'Model loaded on {device}') 



tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b-instruct')

gen_text = transformers.pipeline(
    model=fmodel, 
    tokenizer=tokenizer, 
    task='text-generation', 
    return_full_text=True, 
    device=device, 
    max_length=10000, 
    temperature=0.1, 
    top_p=0.15, #select from top tokens whose probability adds up to 15%
    top_k=0, #selecting from top 0 tokens 
    repetition_penalty=1.1, #without a penalty, output starts to repeat 
    do_sample=True, 
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)


result = gen_text("What is the name of the first president of the united arab emirates?") 
print(result[0]['generated_text']) 

def csv_tool(filename : str):

    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(gen_text(temperature=0), df, verbose=True)
    # return create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.
    Args:
        agent: The agent to query.
        query: The query to ask the agent.
    Returns:
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 
        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
        3. If a line chart is more appropriate, your reply should look like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
        Note: We only accommodate two types of charts: "bar" and "line".
        4. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}
        For example:
           {"answer": "The Product with the highest Orders is '15143Exfo'"}
        5. If the answer is not known or available, respond with:
           {"answer": "I do not know."}
        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}
        Now, let's tackle the query step by step. Here's the query for you to work on: 
        """
        + query
    )

    # Run the prompt through the agent and capture the response.
    response = agent.run(prompt)

    # Return the response converted to a string.
    return str(response)

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.
    Args:
        response (str): response from the model
    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.
    Args:
        response_dict: The response from the agent.
    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                    col: [x[i] if isinstance(x, list) else x for x in data['data']]
                    for i, col in enumerate(data['columns'])
                }       
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

# Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")


    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
st.set_page_config(page_title="👨‍💻 Chat with Data")
st.title("👨‍💻 Chat with Data")

st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload a CSV" , type="csv")

query = st.text_area("Send a Message")

if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
    agent = csv_tool(data)

    # Query the agent.
    response = ask_agent(agent=agent, query=query)

    # Decode the response.
    decoded_response = decode_response(response)

    # Write the response to the Streamlit app.
    write_answer(decoded_response)