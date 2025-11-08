from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.auto_login import get_site_comb_from_filepath
from mcp.server.fastmcp import FastMCP
import asyncio
import simplejson as json
import os
import tempfile
import subprocess
import numpy as np
import base64
import pickle
import dill
import ast
import gzip

mcp = FastMCP("web")
global_env = None

def dict_to_string(data):
    """Convert dict with numpy array to JSON string"""
    serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Convert numpy array to base64 string
            serializable[key] = {
                '_type': 'ndarray',
                'data': base64.b64encode(value.tobytes()).decode('utf-8'),
                'shape': value.shape,
                'dtype': str(value.dtype)
            }
        else:
            serializable[key] = value

    return json.dumps(serializable)

@mcp.tool()
async def login(config_file) -> str:
    try:
        config_file = 'agent_scripts/webarena-agent/webarena/' + config_file
        # get intent
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]
            task_id = _c["task_id"]
            # automatically login
            if _c["storage_state"]:
                cookie_file_name = os.path.basename(_c["storage_state"])
                comb = get_site_comb_from_filepath(cookie_file_name)
                temp_dir = tempfile.mkdtemp()
                # subprocess to renew the cookie
                subprocess.run(
                    [
                        "python",
                        "agent_scripts/webarena-agent/webarena/browser_env/auto_login.py",
                        #"--auth_folder",
                        #temp_dir,
                        "--site_list",
                        *comb,
                    ]
                )
                _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                assert os.path.exists(_c["storage_state"])
                # update the config file
                config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                with open(config_file, "w") as f:
                    json.dump(_c, f)
    except Exception as e:
        return f'Error logging in: {str(e)}'
    return f"Logged in {config_file}"

#@mcp.tool()
#async def env_step(action):

@mcp.tool()
async def get_page_env() -> str:
   global global_env
   page = global_env.page
   #return str(page)
   page = dill.dumps(page)
   return page


@mcp.tool()
async def get_client_env() -> str:
   global global_env
   client = global_env.get_page_client(global_env.page)
   client = str(pickle.dumps(client))
   return client

@mcp.tool()
async def close_env() -> str:
    """
    Close the environment
    """
    global global_env
    def _close():
        return global_env.close()
    
    await asyncio.to_thread(_close)

    return "Environment closed"



@mcp.tool()
async def step_env(action: str) -> str:
    """
    Perform an action on the environment
    """
    global global_env
    #action = json.loads(action)
    action = pickle.loads(ast.literal_eval(action))
    #return f'Unpickled action: {action}'

    def _step(action):
        return global_env.step(action)

    obs, arg2, terminated, arg4, info = await asyncio.to_thread(_step, action)
    obs['image'] = None
    page = info['page']
    info['page'] = None

    page_dict = {
            "url": page.url,
            "content": page.content
    }

    obs = json.dumps(obs)
    obs = gzip.compress(obs.encode('utf-8'))
    obs = base64.b64encode(obs).decode('ascii')

    info = json.dumps(info)
    info = gzip.compress(info.encode('utf-8'))
    info = base64.b64encode(info).decode('ascii')

    page = json.dumps(page_dict)
    page = gzip.compress(page.encode('utf-8'))
    page = base64.b64encode(page).decode('ascii')

    #terminated = json.dumps(terminated)
    #terminated = "terminated"
    #arg2 = json.dumps(arg2)
    #arg4 = json.dumps(arg4)
    #obs = "obs"
    #info = "info"
    arg2 = "arg2"
    arg4 = "arg4"
    #page = "page"
    return f'{obs} ---=--- {arg2} ---=--- {terminated} ---=--- {arg4} ---=--- {info} ---=--- {page}'




@mcp.tool()
async def reset_env(config_file: str) -> str:
    """
    Reset the environment
    """
    global global_env
    
    # Run the sync reset in a separate thread
    def _reset():
        return global_env.reset(
            options={"config_file": 'agent_scripts/webarena-agent/webarena/' + config_file}
        )
    
    obs, info = await asyncio.to_thread(_reset)
    obs['image'] = None
    page = info['page']
    info['page'] = None

    page_dict = {
            "url": page.url,
            "content": page.content
    }
    #obs = dict_to_string(obs)
    #info = dict_to_string(info)
    #obs = pickle.dumps(obs)
   # obs = base64.b64encode(obs).decode('ascii')
    #info = pickle.dumps(info)#.encode('base64', 'strict') 
    obs = json.dumps(obs)
    info = json.dumps(info)
    page = json.dumps(page_dict)
    return f'{obs} ---=--- {info} ---=--- {page}'

@mcp.tool()
async def init_env(render: bool, slow_mo: int, observation_type: str,
                   current_viewport_only: bool,
                   viewport_width: int, viewport_height: int,
                   save_trace_enabled: bool, sleep_after_execution: float) -> str:
    """
    Initialize the browser environment for agent interactions
    """
    global global_env
    
    # Run the sync initialization in a separate thread
    def _init():
        return ScriptBrowserEnv(
            headless=not render,
            slow_mo=slow_mo,
            observation_type=observation_type,
            current_viewport_only=current_viewport_only,
            viewport_size={
                "width": viewport_width,
                "height": viewport_height,
            },
            save_trace_enabled=save_trace_enabled,
            sleep_after_execution=sleep_after_execution,
        )
    
    global_env = await asyncio.to_thread(_init)
    return f'Env initialized successfully'

if __name__ == "__main__":
    mcp.run(transport='stdio')
