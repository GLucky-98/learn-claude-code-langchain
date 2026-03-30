import os
import subprocess
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain.tools import tool
from chat_history_viewer import messages_to_json

# Step1: load env variables
load_dotenv(override=True)
MODEL='MiniMax-M2.7'


# Step2: get a llm agent
lc_agent=init_chat_model(model=MODEL,
                       api_key=os.getenv("MINIMAX_API_KEY"),
                       base_url=os.getenv("MINIMAX_BASE_URL"),
                       model_provider='anthropic',
                       )


# Step3: define system_prompt
System_prompt=SystemMessage(content="You are a smart and professional personal assistant")

# Step4: define tools
tools=[]

@tool
def run_bash(command:str) -> str:
    """
    Execute a bash command and return the output
    """
    # safety check
    for c in ['rm']:
        if c in command:
            return "Error:dangerous command"
    try:
        result=subprocess.run(command,
                              shell=True,
                              cwd=os.getcwd(),
                              capture_output=True,
                              text=True,
                              check=True)
        result=(result.stdout+result.stderr).strip()
        print(result)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Tool invoke Error:run_bash error {e.stderr}")
        return f"Error:{e.stderr}"

tools.append(run_bash)


# Step5: bind tools
lc_agent=lc_agent.bind_tools(tools)


# Step6: def agent loop
def agent_loop(messages):
    while True:
        response=lc_agent.invoke(messages)
        messages.append(response)
        
        # if end turn exit loop
        if response.response_metadata["stop_reason"]=='end_turn':
            print(response.usage_metadata['input_tokens'])
            return messages
        
        # process tool invocation
        if response.tool_calls:
            print("--"*20,"tool_use","--"*20)
            for tool in response.tool_calls:
                try:
                    tool_result=run_bash.invoke(tool)
                    messages.append(tool_result)
                    print("--"*20,tool_result,"--"*20)
                except Exception as e:
                    print(f"Error:{e}")
                    messages.append(ToolMessage(content=f"Error:{e}",tool_call_id=tool["id"]))




# Step7: run agent
if __name__ == "__main__":
    messages=[System_prompt]
    while True:
        try:
            user_input=input("USER:>>")
        except Exception as e:
            print(f"error:{e}")
            break
        if user_input.strip() in ['q','quit']:
            break
        messages.append(HumanMessage(content=user_input))
        agent_loop(messages)
        if hasattr(messages[-1],'text'):
            print(messages[-1].text)
    


    messages_to_json(messages)
    
