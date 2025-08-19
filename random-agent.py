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

#Set up Logging
logger = logging.getLogger(__name__)
path = f"./experiments/{grid}-random"
timestr = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(path, exist_ok=True)
logging.basicConfig(filename=f"{path}/{timestr}-llm-grid-agent.log", level=logging.INFO)

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

def make_gridmap_obs_llm_friendly(obs):
    position = np.argwhere(obs == 'R')
    obs_seen = np.array([''.join(quest_row).replace(' ','.') for quest_row in np.reshape([char for row in obs for char in row], (len(obs), len(obs[0]))).tolist()])
    return obs_seen, position

def random_get_action(obs, last_action, last_reward, last_position, iteration, run):
    global chat_history
    cur_grid, cur_pos = make_gridmap_obs_llm_friendly(obs)
    
    action_ref = { None: "This is the start position, so no action was taken yet.",
                    0: "0: \"move down\"",
                    1: "1: \"move right\"",
                    2: "2: \"move up\"",
                    3: "3: \"move left\"",
                    -1: "-1: Error: no movement",
                  }
    '''
    reward_dict = {None: "This is the start position, so no reward.",
               -0.01: "No note.",
               -0.05: "No note.",
               -0.1: "You collided with a wall. Do not repeat this action and do not cause further collisions.",
               -0.2: "You did not format the output correctly. Check future output formats!",
               1.0: "You found the goal, great job",
              }
    current_status = f"""### Iteration {iteration}
- prev → curr  : {last_position} -> {cur_pos}  
- action       : {last_action}  
- reward       : {last_reward}: {reward_dict[last_reward]}
<observation>
{cur_grid}
</observation>
"""
    print(current_status)
    logging.info(current_status)
    '''
    action = random.randint(0,3)
    action_dict= {'direction_int': action, 'direction_str': action_ref[action]}    
    return cur_pos, action_dict['direction_int'], action_dict['direction_str']

def main():
    global grid, GRIDWORLD
    py_env = create_environment()
    grid_world,_ = make_gridmap_obs_llm_friendly(GRIDWORLD)
    print(f"[ENV-Info] Gridworld {grid} selected:\n{grid_world}")
    logging.info(f"[ENV-Info] Gridworld {grid} selected:\n{grid_world}")

    p_exploration_hist = []

    n_runs = 500
    for cur_run in np.arange(n_runs):
        obs = py_env.reset()
        done = False
        total_reward = 0
        num_steps = 0
    
        last_position = None
        last_action = None
        last_reward = None
        while not done:
            current_position, action_num_val, action_str = random_get_action(obs, last_action, last_reward, last_position, num_steps, cur_run)
            last_position = current_position
            obs, reward, terminated, _ = py_env.step(action_num_val)
            #print(f"[Iteration {str(num_steps).zfill(4)}] Action: {action_num_val} \"{action_str}\" was selected. Reward: {reward}, goal reached: {terminated}")
            #logging.info(f"[Iteration {str(num_steps).zfill(4)}] Action: {action_num_val} \"{action_str}\" was selected. Reward: {reward}, goal reached: {terminated}")
            last_action = (action_num_val, action_str)
            last_reward = reward
            
            total_reward += reward
            num_steps += 1
            done = num_steps > 400 or terminated
        p_explored = py_env.get_reveal_percent()
        p_exploration_hist.append(p_explored)
        cur_grid, cur_pos = make_gridmap_obs_llm_friendly(obs)
        result_msg = f"[RESULT] In run {cur_run} after {num_steps} Iterations the map was revealed to {(p_explored * 100):.2f} %\n{cur_grid}"
        print(result_msg)
        logging.info(result_msg)
    np_p_explored = np.array(p_exploration_hist)
    avg_explore = np.average(np_p_explored)
    std_explore = np.std(np_p_explored)
    median_explore = np.median(np_p_explored)
    min_explore = np.min(np_p_explored)
    max_explore = np.max(np_p_explored)

    end_result_msg = f"[FINAL RESULT] After {n_runs} runs the map was explored: avg.: {(avg_explore * 100):.2f}% (std {(std_explore* 100):.2f}%), median: {(median_explore * 100):.2f}%, min: {(min_explore * 100):.2f}%, max: {(max_explore * 100):.2f}% based on following data:\n{np_p_explored}"
    print(end_result_msg)
    logging.info(end_result_msg)

if __name__ == "__main__":
    main()
