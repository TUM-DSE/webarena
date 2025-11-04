from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web")

global_env = None

@mcp.tool()
async def init_env(render: bool, slow_mo: bool, observation_type: str,
                   current_viewport_only: bool, 
                   viewport_width: int, viewport_height: int,
                   save_trace_enabled: bool, sleep_after_execution: float) -> str:
    """
    Initialize the broswer environment for agent interactions
    """
    global global_env
    global_env = ScriptBrowserEnv(
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

    return f'Env initialized successfully'

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')


