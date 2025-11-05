"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
import ast
import numpy as np
import base64
import pickle

import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

from multiprocessing import shared_memory
from guardian_api import llm
from guardian_api import extract_MCP
from guardian_api import call_MCP
from guardian_api import get_tools


class Context:
    def __init__(self):
        self.shm_to_guardian = None
        self.shm_from_guardian = None
        self.shm_flags = None
        self.request_flag = 0
        self.response_flag = 1


LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="completion")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def generate_init_env_mcp(args):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_server__init_env", 
            "arguments": {
                "render": args.render, 
                "slow_mo": args.slow_mo, 
                "observation_type": args.observation_type, 
                "current_viewport_only": args.current_viewport_only, 
                "viewport_width": args.viewport_width, 
                "viewport_height": args.viewport_height, 
                "save_trace_enabled": args.save_trace_enabled, 
                "sleep_after_execution": args.sleep_after_execution
            }
        }
    }
    msg = json.dumps(msg)
    print(msg)
    return msg

def generate_reset_env_mcp(config_file):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_server__reset_env", 
            "arguments": {
                "config_file": config_file
            }
        }
    }
    msg = json.dumps(msg)
    print(msg)
    return msg


def generate_login_mcp(config_file):
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_server__login", 
            "arguments": {
                "config_file": config_file
            }
        }
    }
    msg = json.dumps(msg)
    print(msg)
    return msg


def string_to_dict(json_string):
    """Reconstruct dict with numpy array from JSON string"""
    data = json.loads(json_string)
    reconstructed = {}
    
    for key, value in data.items():
        if isinstance(value, dict) and value.get('_type') == 'ndarray':
            # Reconstruct numpy array from base64 string
            array_data = base64.b64decode(value['data'])
            array = np.frombuffer(array_data, dtype=value['dtype'])
            reconstructed[key] = array.reshape(value['shape'])
        else:
            reconstructed[key] = value
    
    return reconstructed


def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
    context
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    init_env_msg = generate_init_env_mcp(args)
    print('Calling MCP env init')
    result = call_MCP(context, init_env_msg)
    print(f'[Agent] Got: {result}')


    
    #env = ScriptBrowserEnv(
    #    headless=not args.render,
    #    slow_mo=args.slow_mo,
    #    observation_type=args.observation_type,
    #    current_viewport_only=args.current_viewport_only,
    #    viewport_size={
    #        "width": args.viewport_width,
    #        "height": args.viewport_height,
    #    },
    #    save_trace_enabled=args.save_trace_enabled,
    #    sleep_after_execution=args.sleep_after_execution,
    #)

    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            login_mcp = generate_login_mcp(config_file)
            print('Calling MCP login')
            result = call_MCP(context, login_mcp)
            print(f'[Agent] Got login result: {result}')

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # automatically login
                #if _c["storage_state"]:
                #    cookie_file_name = os.path.basename(_c["storage_state"])
                #    comb = get_site_comb_from_filepath(cookie_file_name)
                #    temp_dir = tempfile.mkdtemp()
                #    # subprocess to renew the cookie
                #    # TODO: Do I need to do this on the server?
                #    subprocess.run(
                #        [
                #            "python",
                #            "browser_env/auto_login.py",
                #            "--auth_folder",
                #            temp_dir,
                #            "--site_list",
                #            *comb,
                #        ]
                #    )
                #    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                #    assert os.path.exists(_c["storage_state"])
                #    # update the config file
                #    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                #    with open(config_file, "w") as f:
                #        json.dump(_c, f)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            print('[TEO] >>> Reset agent conf file')
            trajectory: Trajectory = []
            # TODO: Call guardian to reset env
            #obs, info = env.reset(options={"config_file": config_file})
            reset_mcp = generate_reset_env_mcp(config_file)
            print('Calling MCP reset init')
            result = call_MCP(context, reset_mcp)
            result = json.loads(result)["content"]
            result = result.split('---=---')
            print(f"Raw obs: {result[0]}")
            obs = json.loads(result[0])
            print(f"!!! [TEOO] Obs: {type(obs)}")
            #info = string_to_dict(result[1])
            info = result[1]
            print(f"!!! [TEOO] Info: {type(info)}")
            print(f'[Agent] Got obs: {obs}')
            print(f'[Agent] Got info: {info}')
            print(f'[TEO] >>> Reset env')
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            while True:
                print('[TEO] >>> Test early stop action')
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )
                print('[TEO] >>> Tested early stop action')

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        print('[TEO] >>> Generating next action')
                        action = agent.next_action(
                            trajectory, intent, meta_data=meta_data
                        )
                        print('[TEO] >>> Generated next action')
                    except ValueError as e:
                        print("[TEO] >>>> GOT error message: {str(e)}")
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)
                print(f"Generated action: {action_str}")

                if action["action_type"] == ActionTypes.STOP:
                    print(f'[TEO] Action type was STOP')
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    # TODO: Implement
    env.close()
    if len(scores) != 0:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info('Average score: 0 (no scores)')


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0
    prepare(args)

    # TODO: Get id from guardian
    my_id = 42
    # TODO: Move in agent runtime
    context = Context()
    context.shm_to_guardian = shared_memory.SharedMemory(name=f"shm_to_guard_{my_id}")
    context.shm_from_guardian = shared_memory.SharedMemory(name=f"shm_from_guard_{my_id}")
    context.shm_flags = shared_memory.SharedMemory(name=f"shm_flags_{my_id}")



    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        agent = construct_agent(args)
        test(args, agent, test_file_list, context)
