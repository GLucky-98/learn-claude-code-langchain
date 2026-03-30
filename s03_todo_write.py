import os
import subprocess
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from chat_history_viewer import messages_to_json
from pathlib import Path
from typing import List,Literal
from pydantic import BaseModel,Field

# load env variables
load_dotenv(override=True)
MODEL='MiniMax-M2.7'


# get a llm agent
lc_agent=init_chat_model(model=MODEL,
                       api_key=os.getenv("MINIMAX_API_KEY"),
                       base_url=os.getenv("MINIMAX_BASE_URL"),
                       model_provider='anthropic',
                       )


# define system_prompt
System_promt=SystemMessage(content="You are a smart and professional personal assistant who plans tasks before executing them and follows the plan step by step")

# define tools
tools=[]

# define the task type
class Task(BaseModel):
    id:str = Field(description="unique task identifier")
    content: str = Field(description="task content")
    status: Literal["pending","in_progress","completed"] = Field(description="the status of the task")

class TaskList(BaseModel):
    tasks: List[Task] = Field(description="the task list")

# define a TaskManager
class TaskManager:
    def __init__(self):
        self.tasks=[]
    
    def manager_task_list(self,tasks:List[Task]):
        valid=[]
        for index,task in enumerate(tasks):
            if not task.id:
                task.id=str(f"task:{index+1}")
            if not task.content:
                raise ValueError(f"The task {task.id} is empty")
            if task.status not in ["pending","in_progress","completed"]:
                raise ValueError(f"The status of task {task.id} is invalid ")
            valid.append({'id':task.id,'content':task.content,'status':task.status})
        self.tasks=valid
        return f"Successfully updated the task list!"
    
# instanciate a TaskManager 
task_manager=TaskManager()

# construct the tool function from instance method with args schema
task_manager_tool = StructuredTool.from_function(
                    func=task_manager.manager_task_list,
                    name="task_manager_tool",
                    description="Make a todo list and manager its state",
                    args_schema=TaskList,
                )

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
    # safe command check
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
        return result
    except subprocess.CalledProcessError as e:
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

tools=[run_bash,run_read,run_write,run_edit,task_manager_tool]

# bind tools
lc_agent=lc_agent.bind_tools(tools)

# define a tool execute function
tools_dict={"run_bash":run_bash,"run_read":run_read,"run_write":run_write,"run_edit":run_edit,"task_manager_tool":task_manager_tool}

def execute_tool(tool_call:dict) -> SystemMessage | ToolMessage:     
    tool_name=tool_call["name"]
    tool= tools_dict.get(tool_name,None)
    if tool: # if any avaliable tool
        try:
            tool_result=tool.invoke(tool_call)
            return tool_result
        except Exception: # error during tool execution
            return ToolMessage(content="error during tool execution, check the tool_call content", tool_call_id=tool_call["id"])
    else:
        return ToolMessage(content="No avaliable tool,check the tool_call", tool_call_id=tool_call["id"])

# def agent loop
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
            # print("--"*20,"tool_use","--"*20)
            for tool_call in response.tool_calls:
                tool_result=execute_tool(tool_call)
                messages.append(tool_result)

# run agent
if __name__ == "__main__":
    messages=[System_promt]
    while True:
        try:
            user_input=input("USER:>>")
        except Exception as e:
            print(f"error:{e}")
            break
        if user_input.strip() in ['q','quit']:
            break
        if user_input.strip() == '/task':
            print(task_manager.tasks)
            continue
        messages.append(HumanMessage(content=user_input))
        agent_loop(messages)
        if hasattr(messages[-1],'text'):
            print(messages[-1].text)
    

    messages_to_json(messages)