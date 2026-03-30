import os
import subprocess
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain.tools import tool
from chat_history_viewer import messages_to_json
from pathlib import Path

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
System_prompt=SystemMessage(content="你是一个专业的个人助手")

# Step4: define tools
tools=[]

# safe sandbox
def safe_path(filepath:str) -> Path | str:
    parent=os.getcwd()
    child=Path(filepath).resolve()
    if child.is_relative_to(parent):
        return child
    else:
        return "the path is not allowed"

# bash tool
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
        # print(result)
        return result
    except subprocess.CalledProcessError as e:
        # print(f"Tool invoke Error:run_bash error {e.stderr}")
        return f"Error:{e.stderr}"

# read file
@tool
def run_read(path: str, limit: int = None) -> str:
    """
    read a file and return the content
    """   
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

# write file 
@tool
def run_write(path: str, content: str) -> str:
    """
    write content to a file 
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

# edit file
@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    replace the file's content
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

tools=[run_bash,run_read,run_write,run_edit]

# Step5: bind tools
lc_agent=lc_agent.bind_tools(tools)

# define a tool execute function
tools_dict={"run_bash":run_bash,"run_read":run_read,"run_write":run_write,"run_edit":run_edit}
def execute_tool(tool_call:dict) -> ToolMessage: 
    tool_name=tool_call["name"]
    tool= tools_dict.get(tool_name,None)
    if tool: # if any avaliable tool
        try:
            tool_result=tool.invoke(tool_call)
            return tool_result
        except: # error during tool execution
            return ToolMessage(content="error during tool execution, check the tool_call content", tool_call_id=tool_call["id"])
    else:
        return ToolMessage(content="No avaliable tool,check the tool_call", tool_call_id=tool_call["id"])

# Step6: def agent loop
def agent_loop(messages):
    while True:
        response=lc_agent.invoke(messages)
        messages.append(response)
        
        # if end turn exit loop
        if response.response_metadata["stop_reason"]=='end_turn':
            print(response.usage_metadata['total_tokens'])
            return messages
        
        # process tool invocation
        if response.tool_calls:
            print("--"*20,"tool_use","--"*20)
            for tool_call in response.tool_calls:
                tool_result=execute_tool(tool_call)
                messages.append(tool_result)

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
