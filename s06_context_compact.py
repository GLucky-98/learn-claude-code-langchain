import os
import subprocess
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from chat_history_viewer import messages_to_json
from pathlib import Path
import re
from pydantic import BaseModel,Field
from typing import List

# load env variables
load_dotenv(override=True)
MODEL='MiniMax-M2.7'

# get a llm agent
lc_agent=init_chat_model(model=MODEL,
                       api_key=os.getenv("MINIMAX_API_KEY"),
                       base_url=os.getenv("MINIMAX_BASE_URL"),
                       model_provider='anthropic',
                       )

# define skill
class AgentSkill:
    def __init__(self, skill_dir:str = './skills'):
        self.skills={}
        self.skill_dir=Path(skill_dir)
        for f in self.skill_dir.rglob("SKILL.md"):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get('name',f.parent.name)
            description=meta.get('description',None)
            self.skills[name] = {'name':name, 'description':description, 'body':body, 'path':str(f.resolve())}
        self.skillprompt='\n'.join([f"skill name:{skill['name']} , skill description:{skill['description']}" for _,skill in self.skills.items()])

    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        lines=match.group(1).strip().splitlines()
        name_line=lines[0]
        description_line=''.join(lines[1:])
        meta['name']=name_line.split(':',1)[1].strip()
        meta['description']=description_line.split(':',1)[1].strip()
        return meta, match.group(2).strip()

    def load_skill(self,name:str):
        skill=self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        else:
            return f"skill name: '{name}', skill path: '{skill['path']}', skill content: '{skill['body']}'"

skills=AgentSkill()

# define system_prompt
System_prompt=SystemMessage(content=f"You are a smart and professional personal assistant , your avaliable skills:{skills.skillprompt} , when you need a skill you can use load_skill tool to obtain the specific usage method of skill")

# define tools
tools=[]

# define load skill tool
class skill(BaseModel):
    name:str

load_skill = StructuredTool.from_function(
                    func=skills.load_skill,
                    name="load_skill",
                    description="Obtain the specific usage method of skill",
                    args_schema=skill,
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

tools=[run_bash,run_read,run_write,run_edit,load_skill]

# bind tools
lc_agent=lc_agent.bind_tools(tools)

# define a tool execute function
tools_dict={"run_bash":run_bash,"run_read":run_read,"run_write":run_write,"run_edit":run_edit,"load_skill":load_skill}

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

# define context compact function tool
def context_compatc(messages:List[BaseMessage]):
    
    sub_mesaages=messages+[HumanMessage(content="summary the conversation, conserve the important content")]
    response=lc_agent.invoke(sub_mesaages)
    
    # Compare the conversation before and after compation
    # the agent sometimes will use the write tool to save the important things
    messages_to_json(messages)
    print(f"compact completed,compact rate:{response.usage_metadata['output_tokens']/response.usage_metadata['input_tokens']}")
    
    # get new compacted conversation
    del messages[1:]
    messages.append(response)
    return messages

# def agent loop
def agent_loop(messages):
    while True:
        response=lc_agent.invoke(messages)
        messages.append(response)
        
        # if end turn exit loop
        if response.response_metadata["stop_reason"]=='end_turn':
            print(response.usage_metadata['total_tokens'])
            if len(messages)>5: # just for example, it's not for practice
                messages=context_compatc(messages)
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
