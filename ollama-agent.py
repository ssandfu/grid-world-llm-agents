import numpy as np
import csv
import time
import random
import os

import logging

from environment_5x5_exploration import environment_5x5
from openai import OpenAI
from pydantic import BaseModel
import ollama

#######################
# HYPERPARAMETERS     #
#######################

# Grid-world map (19x21 characters, walls='W', goal='g', etc.)
GRIDWORLD = [
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
START_ROW = 17
START_COL = 10

gw_line_width_np = 2 + (len(GRIDWORLD[0])+1) + 2
np.set_printoptions(linewidth=gw_line_width_np)

model_id = 'deepseek-r1:70b'#'deepseek-r1:32b'#'gpt-oss:20b'#'gpt-oss:120b'#'mixtral'
#model_id = 'gpt-4.1'
if model_id in ["gpt-4.1","o3"]:
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    top_level_role = "developer"
else:
    client = OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )
    top_level_role = "system"

#Set up Logging
logger = logging.getLogger(__name__)
path = f"./experiments/{model_id}"
timestr = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(path, exist_ok=True)
logging.basicConfig(filename=f"{path}/{timestr}-llm-grid-agent.log", level=logging.INFO)

class ActionOutput(BaseModel):
  direction: str
  numericVal: int

def create_environment():
    """Create a new 5x5 grid-world environment instance.

    Returns:
        environment_5x5: custom Gym environment class
    """
    return environment_5x5(
        gridworld=GRIDWORLD,
        start_row=START_ROW,
        start_col=START_COL,
        max_steps=1000
    )

task_explain_navigation = "You are a helpful navigation assistant that has to help a robot get to his goal destination. Do not crash into the wall and navigate the robot to the goal position. You have to give an action to move the robot based on the current observation and previous observations in your context window. "

task_explain_exploration = "Your task is to help a robot explore an unknown environment displayed as a gridworld and uncover all of the unknown tiles inside a gridworld. You have to give an action to move the robot based on the current observation and previous observations in your context window."

observation_explain_5x5 = "As an observation you get a 5x5 grid around your current position as an array of characters that describes the current grid around the robot. The position that the robot is on is in the center of the 5x5 grid that is given as an oberservation. "

observation_explain_exploration = "As an observation you get the grid world with all unexplored tiles and all the revealed tiles with the current position of the robot marked with \"R\". "

grid_world_explain = "The gridworld has its 0,0 coordinate at the top left corner of the grid. The position is the index of rows and columns in the grid world with a base of 0. The gridworld has the dimensions 19 rows and 21 columns. The gridworld is encoded so a \"W\" is a wall, \".\" and \" \" are free cells, \"-\" is a negative cell, \"+\" is a positive cell and \"g\" is the goal. Unknown spaces for exploration are marked with \"?\". "

actions_explained = "The actions have a numerical value that lets the robot move in a cardinal direction. Action 0 is a move down i.e. a single positive movement in the rows direction, Action 1 is a move to the right i.e. a single positive movement in the columns direction, Action 2 is a move up i.e. a single negative movement in the rows direction and Action 3 is a move to the left i.e. a single negative movement in the columns direction. Under no circumstances select an action that would result in a collision with a wall (W). "

#Use these as alternatives when constructing the system prompt
north_goal_information = "The goal is somewhere north of the robot. "
no_goal_information = "Unless you can see it, you do not have any information where the goal position is. "
goal_information = "The goal is at position (1, 10)."

expected_output_message_formated_output = "Return a short summary of your thought progress and the action the robot should take next in the following format: direction: \"<direction in [down, right, up, left]>\", numerical Action Value: <integer in [0,1,2,3]."

history_context = " \n\nBelow you will have a history of previous all observations, the actions you have taken based on the observations and the reward the environment gave you. A higher reward is more desirable than a smaller reward i.e. an action with a reward of -0.01 was still a better action than -0.1 despite being negative. Use primarily the last observation as well as the previous observations, actions and reward to determine the best possible action for the robot. Keep track of where you have already been in the gridworld based on the observations. "

few_shot_no_walls_1="""
Under no circumstances select an action that would result in a robot position inside a wall. In this example do not move downwards, because this would be a collision with a wall. The only allowed action in this example are move up, left or right:
[['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' ' ' ' ' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '-' 'W' 'W' 'W' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' ' ' ' ' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' 'W' 'W' ' ' 'W' 'W' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' ' ' ' ' 'R' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' 'W' 'W' 'W' 'W' 'W' '?' '?' '?' '?' '?' '?' '?' '?']]
 """
few_shot_no_walls_2="""
Under no circumstances select an action that would result in a robot position inside a wall. In this example do not move left or right, because this would be a collision with a wall. The only allowed action in this example are move up or down:
[['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' ' ' ' ' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '-' 'W' 'W' 'W' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' ' ' ' ' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' 'W' 'W' 'R' 'W' 'W' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' ' ' ' ' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' 'W' 'W' 'W' 'W' 'W' '?' '?' '?' '?' '?' '?' '?' '?']]
 """
few_shot_no_walls_3="""
Under no circumstances select an action that would result in a robot position inside a wall. In this example do not move up or down, because this would be a collision with a wall. The only allowed action in this example are move left or right:
[['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' ' ' ' ' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' '?' '-' 'W' 'W' 'W' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' 'W' ' ' ' ' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' 'W' 'W' 'W' ' ' 'W' 'W' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' ' ' ' ' 'R' ' ' ' ' ' ' '?' '?' '?' '?' '?' '?' '?' '?']
 ['?' '?' '?' '?' '?' '?' '?' 'W' 'W' 'W' 'W' 'W' 'W' '?' '?' '?' '?' '?' '?' '?' '?']]
 """


prompt_experiment_base = task_explain_navigation + grid_world_explain + observation_explain_5x5  + \
                         actions_explained + north_goal_information + \
                         expected_output_message_formated_output + history_context

prompt_experiment_base_goal = task_explain_navigation + grid_world_explain + observation_explain_5x5  + \
                         actions_explained + goal_information + \
                         expected_output_message_formated_output + history_context

prompt_experiment_exploration = task_explain_exploration + grid_world_explain + observation_explain_exploration  + \
                         actions_explained + expected_output_message_formated_output + \
                         few_shot_no_walls_1 + few_shot_no_walls_2 + \
                         few_shot_no_walls_3 + history_context

prompt_experiment_exploration_0_shot = task_explain_exploration + grid_world_explain + observation_explain_exploration  + \
                                        actions_explained + expected_output_message_formated_output + history_context

system_message = {'role': top_level_role, 'content': prompt_experiment_exploration_0_shot}

chat_history = [system_message]
print(chat_history)
logging.info(chat_history)

movement_history = []

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

def interpret_reward_llm_friendly(cur_position, last_action, last_reward):
    reward_dict = {None: "This is the start position, so no reward.",
                   -0.01: "You did everything right and navigated to an empty space.",
                   -0.05: "You found a negative space. These are also viable empty spaces.",
                   -0.1: "You collided with a wall. Do not repeat this action and do not cause further collisions.",
                   1.0: "You found the goal, great job",
                  }
    action_dict = { None: "This is the start position, so no action was taken yet.",
                    0: "0: \"move down\"",
                    1: "1: \"move right\"",
                    2: "2: \"move up\"",
                    3: "3: \"move left\"",
                  }
    movement_message = f"The reward to move from the previous position to {cur_position} using the action {action_dict[last_action]} was given the reward \"{reward_dict[last_reward]}"
    movement_history.append(movement_message)
    message = f"+++ This is the movement history of the current session:\n{movement_history}\n+++ This is the observation at the position {cur_position} which was reached from the previous position using the action {action_dict[last_action]} and was given the reward \"{reward_dict[last_reward]}\".\n"
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


#def llm_structured_output(message, text_format):

def llm_get_action(obs, last_action, last_reward, max_len_context=50):
    global chat_history
    #cur_grid, cur_pos = make_obs_llm_friendly(obs)
    cur_grid, cur_pos = make_gridmap_obs_llm_friendly(obs)
    #cur_grid = obs
    #cur_pos = np.argwhere(cur_grid == 'R')
    context_message = interpret_reward_llm_friendly(cur_pos, last_action, last_reward)
    context_message += f"{cur_grid}\n"
    print(context_message)
    logging.info(context_message)
    chat_history.append({'role':'user', 'content': context_message})
    chat_message = [system_message, chat_history[-1]]
    
    raw_response = chat_llm(chat_message)


    interpretation_message = [system_message, {'role': 'user', 'content': f"This message was given by an LLM to provide an action to the movement of a robot in a gridworld. Please interpret the message an find the action that is suggested. The actions have a numerical value that lets the robot move in a cardinal direction. Action 0 is a move down i.e. a single positive movement in the rows direction, Action 1 is a move to the right i.e. a single positive movement in the columns direction, Action 2 is a move up i.e. a single negative movement in the rows direction and Action 3 is a move to the left i.e. a single negative movement in the columns direction. +++ {raw_response}"}]
    print("\n")
    interpreted_response = chat_llm(interpretation_message, text_format=ActionOutput)#.model_json_schema())#, options={"temperature":0.0})
    print("\n")
    print(interpreted_response)
    action = ActionOutput.model_validate_json(interpreted_response)
    if action.numericVal not in [0,1,2,3]:
        print(f"[WARN]: LLM did not select appropriate action ({action.numericVal}). An action is randomly selected.")
        logging.info(f"[WARN]: LLM did not select appropriate action ({action.numericVal}). An action is randomly selected.")
        action.numericVal = random.randint(0,3)
    #print("[DEBUG]: Full response test. An action is randomly selected.")
    #logging.info("[DEBUG]: Full response test. An action is randomly selected.")
    return raw_response, action.numericVal

def main():
    py_env = create_environment()
    
    obs = py_env.reset()
    done = False
    total_reward = 0
    num_steps = 0

    last_action = None
    last_reward = None
    
    while not done:
        #print(obs)
        raw_output, action = llm_get_action(obs, last_action, last_reward)
        obs, reward, terminated, _ = py_env.step(action)
        print(f"[Iteration {str(num_steps).zfill(4)}] Action: {action} was selected. Reward: {reward}, goal reached: {terminated}")
        logging.info(f"[Iteration {str(num_steps).zfill(4)}] Action: {action} was selected. Reward: {reward}, goal reached: {terminated}")
        last_action = action
        last_reward = reward
        
        total_reward += reward
        num_steps += 1
        done = num_steps > 1000 or terminated
if __name__ == "__main__":
    main()
