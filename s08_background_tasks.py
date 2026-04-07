import os
import subprocess

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from chat_history_viewer import messages_to_json
from pathlib import Path
import threading

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
System_prompt=SystemMessage(content="You are a smart and professional personal assistant")

# define tools
tools=[]

# define backgroundtaskmanager
class BackgroundManager():
    def __init__(self) -> None:
        self.tasks={} #  key 'task_id'  value: {'status','result','command'}
        self._notification_queue=[]
        self._lock=threading.Lock()
    
    def run(self,command:str) -> str:
        with self._lock:
            task_id=str(len(self.tasks)+1)
            self.tasks[task_id]={'status':'running', 'command':command, 'result':None}
        
        thread=threading.Thread(target=self._execute,args=(task_id,command),daemon=True)
        thread.start()
        return f"Background task started, task id: {task_id}"
    
    
    def _execute(self,task_id: str,command: str):
        try:
            result=subprocess.run(command,
                        shell=True,
                        cwd=os.getcwd(),
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=300)
            result=(result.stdout + result.stderr).strip()[:5000] or 'No output'
            status='completed'
        except subprocess.TimeoutExpired:
            result='Error: Timeout (300s)'
            status='timeout'
        except Exception as e:
            result=f'Error: {e}'
            status='Error'

        with self._lock:
            self.tasks[task_id]['result']=result
            self.tasks[task_id]['status']=status
            self._notification_queue.append({'id':task_id, 'status':status, 'command':command, 'result':result})
        
        return 
    
    def check_task(self,task_id: str = None):
        with self._lock:
            if task_id:
                task=self.tasks.get(task_id)
                if not task:
                    return f'Unknown task: {task_id}'
                return f"[{task['status']}], result:{task['result'] or 'running'}"
            
            lines=[]
            for id,task in self.tasks.items():
                lines.append(f"Task {id}, status:{task['status']}, result:{task['result'] or 'running'}")
            
            return '\n'.join(lines) if lines else 'No background task'
    
    def drain_notifications(self):
        with self._lock:
            notifs=list(self._notification_queue)
            self._notification_queue.clear()
        
        lines=[]
        for task in notifs:
            lines.append(f"Task {task['id']}, status:{task['status']}, result:{task['result'] or 'running'}")

        return HumanMessage(content='/n'.join(lines)) if lines else None

BG=BackgroundManager()

run_background_task=StructuredTool.from_function(
                    func=BG.run,
                    name='run_background_task',
                    description='Run a task in the background',
                    args_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
                    )

check_background_task=StructuredTool.from_function(
                    func=BG.check_task,
                    name='check_background_task',
                    description='check the result and status of background task. Input is the task id. If input is None, check all tasks',
                    args_schema={"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}
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

tools=[run_bash,run_read,run_write,run_edit,run_background_task,check_background_task]

# bind tools
lc_agent=lc_agent.bind_tools(tools)

# define a tool execute function
tools_dict={"run_bash":run_bash,
            "run_read":run_read,
            "run_write":run_write,
            "run_edit":run_edit,
            'run_background_task':run_background_task,
            'check_background_task':check_background_task}


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

# def agent loop
def agent_loop(messages):
    while True:
        if notifs:=BG.drain_notifications():
            messages.append(notifs)
            print(notifs)

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

# run agent
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
