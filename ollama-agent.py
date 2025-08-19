import numpy as np
import csv
import time
import random
import os

import logging

from environment_5x5_exploration import environment_5x5
from openai import OpenAI, RateLimitError, InternalServerError
from pydantic import BaseModel
import ollama

import pandas as pd
from pydantic import BaseModel

import json
import re # regex for output filtering

#######################
# HYPERPARAMETERS     #
#######################

# Grid-world map (19x21 characters, walls='W', goal='g', etc.)
GRIDWORLD_HARD = [
    "WWWWWWWWWWWWWWWWWWWWW",
    "W         g         W",
    "W  W W-W-WWW-W WWW  W",
    "W  W W-W     W-W W  W",
    "W  W W-WW   WW   W  W",
    "WW W W-W     W W    W",
    "W    W-W-WWW W W  WWW",
    "WWWW W-W     W WWWWWW",
    "W    W WW   WW    W W",
    "W  WWW W     W WW   W",
    "W  - W W WWW-W W  W W",
    "W  W W W     W WWWW W",
    "W  W W WW   WW-W    W",
    "W  W W W     W-W  WWW",
    "W WW W W-WWW W-W    W",
    "W W  W W     W-W    W",
    "W W WW WWW WWW WW W W",
    "W                   W",
    "WWWWWWWWWWWWWWWWWWWWW",
]
START_HARD = (17,10)

GRIDWORLD_MEDIUM = [
    "WWWWWWWWWWWWWWWWWWWWW",
    "W-------------------W",
    "W-   g             -W",
    "W----------        -W",
    "WWWWWWWWWW-        -W",
    "W----WWWWW-        -W",
    "W-  -WWWWW-        -W",
    "W-  -------        -W",
    "W-                 -W",
    "W-      --------   -W",
    "W-      -WWWWWW--  -W",
    "W-      -WWWWWW--  -W",
    "W-      -WWWWWW--  -W",
    "W-      -WWWWWWWWWWWW",
    "W-      ------------W",
    "W-                 -W",
    "W-                 -W",
    "W-------------------W",
    "WWWWWWWWWWWWWWWWWWWWW",
]
START_MEDIUM = (17,10)

GRIDWORLD_EASY = [
    "WWWWWWWWWWWWWWWWWWWWW",
    "W         g         W",
    "W                   W",
    "W                   W",
    "W                   W",
    "W                   W",
    "W     WWWWWWWWW     W",
    "W     WWWWWWWWW     W",
    "W     WWWWWWWWW     W",
    "W     WWWWWWWWW     W",
    "W     WWWWWWWWW     W",
    "W     WWWWWWWWW     W",
    "W     WWWWWWWWW     W",
    "W                   W",
    "W                   W",
    "W                   W",
    "W                   W",
    "W                   W",
    "WWWWWWWWWWWWWWWWWWWWW",
]
START_EASY = (17,10)

grids = ["easy", "medium", "hard"]
grid = os.getenv("GRIDWORLD", default = "hard")

if grid == "easy":
    GRIDWORLD = GRIDWORLD_EASY
    START_POS = START_EASY
elif grid == "medium":
    GRIDWORLD = GRIDWORLD_MEDIUM
    START_POS = START_MEDIUM
elif grid == "hard":
    GRIDWORLD = GRIDWORLD_HARD
    START_POS = START_HARD
else:
    raise ValueError(f"The Gridworld \"{grid}\" selected is not registered as a valid option. Please select in {grids}.")

START_ROW = START_POS[0]
START_COL = START_POS[1]

gw_line_width_np = 2 + (len(GRIDWORLD[0])+1) + 2
np.set_printoptions(linewidth=gw_line_width_np)

ollama_models = ['gpt-oss:120b', 'gpt-oss:20b', 'deepseek-r1:70b', 'deepseek-r1:32b', 'mixtral']
openai_models = ["gpt-5", "gpt-5-mini"]
gwdg_models = ["gpt-oss-120b", "gemma-3-27b-it", "qwen3-235b-a22b", "qwq-32b", "deepseek-r1", "llama-3.3-70b-instruct", "mistral-large-instruct" ]

model_id = os.getenv("MODEL_ID", default = "gpt-oss-120b")
run_locally = False

#model_id = 'gpt-5'
if ((run_locally==False) and (model_id in openai_models)):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    top_level_role = "developer"
    logging_prefix = "openai"
elif ((run_locally==False) and (model_id in gwdg_models)):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("GWDG_API_KEY"),
        base_url = 'https://chat-ai.academiccloud.de/v1',
    )
    top_level_role = "system"
    logging_prefix = "gwdg"
elif ((run_locally==True) and (model_id in ollama_models)):
    client = OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )
    top_level_role = "system"
    logging_prefix = "ollama"
else:
    raise ValueError(f"The Model \"{model_id}\" selected is not registered as a valid option using the setting run_locally={run_locally}.")

#Set up Logging
logger = logging.getLogger(__name__)
path = f"./experiments/expl-{grid}-{logging_prefix}-{model_id}"
timestr = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(path, exist_ok=True)
logging.basicConfig(filename=f"{path}/{timestr}-llm-grid-agent.log", level=logging.INFO)

print(f"[Model-Info] Selected the Model \"{model_id}\" running on {client.base_url}")
logging.info(f"[Model-Info] Selected the Model \"{model_id}\" running on {client.base_url}")

class ExtractDirection(BaseModel):
    direction_str: str
    direction_int: int

def create_environment():
    """Create a new 5x5 grid-world environment instance.

    Returns:
        environment_5x5: custom Gym environment class
    """
    return environment_5x5(
        gridworld=GRIDWORLD,
        start_row=START_ROW,
        start_col=START_COL,
        max_steps=400,
        use_exploration_mode = True
    )

task_explain_navigation = "You are a helpful navigation assistant that has to help a robot get to his goal destination. Do not crash into the wall and navigate the robot to the goal position. You have to give an action to move the robot based on the current observation and previous observations in your context window. "
#Use these as alternatives when constructing the system prompt
north_goal_information = "The goal is somewhere north of the robot. "
no_goal_information = "Unless you can see it, you do not have any information where the goal position is. "
goal_information = "The goal is at position (1, 10)."

task_explain_exploration = """You are the robot’s navigation module. At each iteration, decide a single cardinal move that will safely uncover new territory, avoid collisions with walls (W), and maximise the number of discovered grid tiles. Continue until every “?” tile has been observed."""

observation_explain_exploration = "As an observation you get the grid world with all unexplored tiles and all the revealed tiles with the current position of the robot marked with \"R\". "

grid_world_explain = """#### Gridworld Reference
'W': Wall (impassable)
'.' or ' ': Free cell
'-': Free cell with negative reward
'g': Goal for navigation tasks
'?': Unexplored tile for exploration tasks
'R': Robot’s current position

*Coordinates* '(row, col)' with origin **(0, 0)** at the **top-left** corner.
"""

actions_explained = """#### Actions:
| Action | Move  | Δ row | Δ col |
|--------|-------|-------|-------|
| '0'    | Down  | +1    | 0     |
| '1'    | Right | 0     | +1    |
| '2'    | Up    | –1    | 0     |
| '3'    | Left  | 0     | –1    |

**Never** choose an action that would put the robot on a space with 'W'."""

expected_output_message_formated_output = """Output the result in the following format: <output>{"direction_str": "<direction>", "direction_int": <int>}</output>.
The valid mappings are: down -> 0, right -> 1, up -> 2, left -> 3.
No other output is allowed.
---
Example:
<output>{"direction_str": "down", "direction_int": 0}</output>
"""

prompt_experiment_exploration_0_shot = task_explain_exploration + grid_world_explain + observation_explain_exploration  + \
                                        actions_explained + expected_output_message_formated_output

system_message = {'role': top_level_role, 'content': prompt_experiment_exploration_0_shot}

chat_history = [system_message]
print(chat_history)
logging.info(chat_history)

movement_history = []
shortened_movement_history = []

def make_obs_llm_friendly(obs):
    trans_dict = {"-1.0": "W",
                  "0.0" : ".",
                  "-0.5": "-",
                  "0.5": "+",
                  "1.0": "g"}
    obs_grid_5x5 = [trans_dict[x] for x in obs[0:25].astype(str).tolist()]
    obs_grid_5x5 = np.reshape(np.array(obs_grid_5x5, dtype=str), (5,5))
    position = np.reshape(obs[25:], (1,2)).astype(int)#.tolist()
    print(obs_grid_5x5) 
    logging.info(obs_grid_5x5)
    return obs_grid_5x5, position

def make_gridmap_obs_llm_friendly(obs):
    position = np.argwhere(obs == 'R')
    obs_seen = np.array([''.join(quest_row).replace(' ','.') for quest_row in np.reshape([char for row in obs for char in row], (len(obs), len(obs[0]))).tolist()])
    return obs_seen, position

def interpret_reward_llm_friendly(cur_position, cur_grid, last_action, last_reward, last_position, iteration):
    reward_dict = {None: "This is the start position, so no reward.",
                   -0.01: "No note.",
                   -0.05: "No note.",
                   -0.1: "You collided with a wall. Do not repeat this action and do not cause further collisions.",
                   -0.2: "You did not format the output correctly. Check future output formats!",
                   1.0: "You found the goal, great job",
                  }
    action_dict = { None: "This is the start position, so no action was taken yet.",
                    0: "0: \"move down\"",
                    1: "1: \"move right\"",
                    2: "2: \"move up\"",
                    3: "3: \"move left\"",
                    -1: "-1: Error: no movement",
                  }
    movement_message_shortened = f"""#Iteration {iteration}:{last_position}->{cur_position} with {last_action} got {last_reward} ({reward_dict[last_reward]})."""
    movement_history.append(movement_message_shortened)
    used_movement_history = movement_history
    '''
    if last_reward in [-0.1, -0.2, 1.0]:
        shortened_movement_history.append(movement_message_shortened)
        print(f"[Info]: Appended iteration {iteration} to shortened movement history.")
        logging.info(f"[Info]: Appended iteration {iteration} to shortened movement history.")
    
    if(len(movement_history) > 30):
        used_movement_history = shortened_movement_history
        used_movement_history.append(movement_history[-20:-1])
        print(f"[Info]: Using shortened movement history with {len(shortened_movement_history)} entries.")
        logging.info(f"[Info]: Using shortened movement history with {len(shortened_movement_history)} entries.")
    else:
        used_movement_history = movement_history
        print(f"[Info]: Using cummulative movement history with {len(shortened_movement_history)} entries.")
        logging.info(f"[Info]: Using cummulative movement history with {len(shortened_movement_history)} entries.")
    '''
    current_status = f"""### Iteration {iteration}
- prev → curr  : {last_position} -> {cur_position}  
- action       : {last_action}  
- reward       : {last_reward}: {reward_dict[last_reward]}
<observation>
{cur_grid}
</observation>
"""
    message_with_history = f"""
+++###Current Status
{current_status}
###Movement History:
{used_movement_history}
"""
    message_no_history = f"""
+++###Current Status
{current_status}
"""
    message = message_no_history
    return message

def chat_llm(messages, text_format=None):
    global client, model_id, chat_history
    if text_format == None:
        response = client.chat.completions.create(
          model=model_id,
          messages=messages
        )
        print(response)
        logging.info(response)
        ret_val = response.choices[0].message.content
        
    else:
        if client.api_key=='ollama':
            response = ollama.chat(model=model_id, messages=messages, format=ActionOutput.model_json_schema())
            print(response)
            logging.info(response)
            ret_val = response.message.content
        else:
            #Caution: This does not generate a response and is only for information extraction!
            response = client.responses.parse(
              model=model_id,
              input=messages,
              text_format=text_format
            )
            print(response)
            logging.info(response)
            ret_val = response.output[0].content[0].text
    chat_history.append({'role': 'assistant','content': ret_val,})
    return ret_val

def chat_structured_output(messages):
    global client, model_id, chat_history
    response = client.chat.completions.parse(
      model=model_id,
      messages=messages,
      response_format=ExtractDirection,
    )
    ret_val = response.choices[0].message.parsed
    chat_history.append({'role': 'assistant','content': ret_val,})
    print(ret_val)
    logging.info(ret_val)
    return ret_val

#def llm_structured_output(message, text_format):

def llm_get_action(obs, last_action, last_reward, last_position, iteration):
    global chat_history
    #cur_grid, cur_pos = make_obs_llm_friendly(obs)
    cur_grid, cur_pos = make_gridmap_obs_llm_friendly(obs)
    #cur_grid = obs
    #cur_pos = np.argwhere(cur_grid == 'R')
    context_message = interpret_reward_llm_friendly(cur_pos, cur_grid, last_action, last_reward, last_position, iteration)
    print(context_message)
    logging.info(context_message)
    chat_history.append({'role':'user', 'content': context_message})
    chat_message = [system_message, chat_history[-1]]

    structured_response = chat_structured_output(chat_message)
    return cur_pos, structured_response.direction_int, structured_response.direction_str
    
    '''
    raw_response = chat_llm(chat_message)

    template = r"<output>(.*?)</output>"
    try:
        action_json = re.search(template, raw_response).group(1)
        action_dict = json.loads(action_json)
    except AttributeError:
        action_dict = {'direction_int': -1, 'direction_str': "No output"}
    print(action_dict)
    logging.info(action_dict)
    return cur_pos, action_dict['direction_int'], action_dict['direction_str']
    '''
    #interpretation_message = [system_message, {'role': 'user', 'content': f"This message was given by an LLM to provide an action to the movement of a robot in a gridworld. Please interpret the message an find the action that is suggested. The actions have a numerical value that lets the robot move in a cardinal direction. Action 0 is a move down i.e. a single positive movement in the rows direction, Action 1 is a move to the right i.e. a single positive movement in the columns direction, Action 2 is a move up i.e. a single negative movement in the rows direction and Action 3 is a move to the left i.e. a single negative movement in the columns direction. +++ {raw_response}"}]
    #print("\n")
    #interpreted_response = chat_llm(interpretation_message, text_format=ActionOutput)#.model_json_schema())#, options={"temperature":0.0})
    #print("\n")
    #print(interpreted_response)
    #action = ActionOutput.model_validate_json(interpreted_response)
    #if action.numericVal not in [0,1,2,3]:
    #    print(f"[WARN]: LLM did not select appropriate action ({action.numericVal}). An action is randomly selected.")
    #    logging.info(f"[WARN]: LLM did not select appropriate action ({action.numericVal}). An action is randomly selected.")
    #    action.numericVal = random.randint(0,3)
    #print("[DEBUG]: Full response test. An action is randomly selected.")
    #logging.info("[DEBUG]: Full response test. An action is randomly selected.")
    #return raw_response, action.numericVal

def main():
    global grid, GRIDWORLD
    py_env = create_environment()
    grid_world,_ = make_gridmap_obs_llm_friendly(GRIDWORLD)
    print(f"[ENV-Info] Gridworld {grid} selected:\n{grid_world}")
    logging.info(f"[ENV-Info] Gridworld {grid} selected:\n{grid_world}")
    
    obs = py_env.reset()
    done = False
    total_reward = 0
    num_steps = 0

    last_position = None
    last_action = None
    last_reward = None
    
    while not done:
        try:#print(obs)
            current_position, action_num_val, action_str = llm_get_action(obs, last_action, last_reward, last_position, num_steps)
            last_position = current_position
            obs, reward, terminated, _ = py_env.step(action_num_val)
            
            #TODO: TASK abhängigkeit einbauen
            p_explored = py_env.get_reveal_percent()
            iteration_msg = f"[Iteration {str(num_steps).zfill(4)}] Action: {action_num_val} \"{action_str}\" was selected. Reward: {reward}, goal reached: {terminated}, explored: {(p_explored * 100):.2f}%"
            print(iteration_msg)
            logging.info(iteration_msg)
            
            last_action = (action_num_val, action_str)
            last_reward = reward
            
            total_reward += reward
            num_steps += 1
            done = num_steps > py_env.max_steps or terminated

            
        
        #When the rate limit is reached, wait for a minute to get into the next rate limit window
        except RateLimitError:
            time.sleep(60)
            print("[WARN] Rate Limit is reached. Waiting for 60 seconds! Then continue with a new iteration.")
            logging.info("[WARN] Rate Limit is reached. Waiting for 60 seconds! Then continue with a new iteration.")
        except InternalServerError:
            time.sleep(300)
            error_msg = "[Error] InternalServerError. Waiting for 5 Minutes!"
            print(error_msg)
            logging.info(error_msg)
    p_explored = py_env.get_reveal_percent()
    cur_grid, cur_pos = make_gridmap_obs_llm_friendly(obs)
    result_msg = f"[RESULT] After {num_steps} Iterations the map was revealed to {(p_explored * 100):.2f} %\n{cur_grid}"
    print(result_msg)
    logging.info(result_msg)
if __name__ == "__main__":
    main()
