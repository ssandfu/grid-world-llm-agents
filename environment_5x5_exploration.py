import sys
import time
import copy
import numpy as np
import gym
import random
from gym import spaces

class environment_5x5(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gridworld, start_row=0, start_col=0, max_steps=100, use_exploration_mode=False):
        # Belohnungen (Rewards) für die Umgebung
        self.wall_reward  = -0.1    # 'W'
        self.minus_reward = -0.05     # '-'
        self.empty_reward = -0.01   # ' '
        self.plus_reward  = 0      # '+'
        self.goal_reward  = 1    # 'g'

        self.no_action_reward  = -0.2
        
        # Robuste, balancierte Kodierung im Bereich [-1, 1]
        self._symbol2idx = {
            'W': -1.0,   # Wand
            '-': -0.5,   # Minuszelle
            ' ':  0.0,   # Leerfeld
            '+':  0.5,   # Pluszelle
            'g':  1.0    # Ziel
        }
        # Stochastische Mauern (Ziffern 1–9), hier einfach als Wände behandelt
        #self._symbol2idx.update({str(i): -1.0 for i in range(1, 10)})
        self._idx2symbol = {v: k for k, v in self._symbol2idx.items()}

        self.world = gridworld
        #Replace all information with ?
        #self.world_seen = [''.join(quest_row) for quest_row in np.reshape(["?" for row in self.world for char in row], (len(self.world), len(self.world[0]))).tolist()]
        self.world_seen = np.reshape(["?" for row in self.world for char in row], (len(self.world), len(self.world[0]))).tolist()
        self.n_rows = len(gridworld)
        self.n_cols = len(gridworld[0])

        self.action_space = spaces.Discrete(4)
        # Beobachtung: 5x5 Patch, Werte zwischen -1 und 1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(27,),
            dtype=np.float32
        )

        self.start_row = start_row
        self.start_col = start_col
        self.max_steps = max_steps
        self.exploration_mode = use_exploration_mode
        self.reset()

    def _get_padded_worldmap(self, padding_width=2):
        grid = np.array([[self._symbol2idx[c] for c in row] for row in self.world])
        padded = np.pad(grid, pad_width=padding_width, mode='constant', constant_values=self._symbol2idx['W'])
        return grid, padded

    def _get_padded_position(self, padding_width=2):
        obs_x, obs_y = self.position
        r_pad = obs_x + padding_width
        c_pad = obs_y + padding_width
        return r_pad, c_pad
    
    def _reveil_map(self, padding_width=2):
        obs_x, obs_y = self.position
        r_pad, c_pad = self._get_padded_position(padding_width=padding_width)
        _, padded = self._get_padded_worldmap(padding_width=padding_width)
        
        r_index = np.arange(r_pad-padding_width, r_pad+padding_width+1)
        c_index = np.arange(c_pad-padding_width, c_pad+padding_width+1)
        #Only access the values that are not padding and are in the original grid
        for r_i in r_index[(r_index >= padding_width) & (r_index < (len(self.world_seen)+padding_width))]:
            #Only access the values that are not padding and are in the original grid
            for c_i in c_index[(c_index >= padding_width) & (c_index < (len(self.world_seen[0])+padding_width))]:
                #Overwrite the question marks in the seen map
                self.world_seen[r_i-padding_width][c_i-padding_width] = self._idx2symbol[padded[r_i, c_i]]
    
    def _get_obs_revealed_map(self, padding_width=2):
        self._reveil_map(padding_width=padding_width)
        world_seen_with_robot_pos = np.array(self.world_seen)
        world_seen_with_robot_pos[self.position] = 'R'
        return world_seen_with_robot_pos

    def get_reveal_percent(self):
        np_world = np.array(self.world_seen)
        n_total = np_world.shape[0] * np_world.shape[1]
        
        unique, counts = np.unique(np_world, return_counts=True)
        try:
            n_unexplored = dict(zip(unique, counts))['?']
        except KeyError:
            n_unexplored = 0

        p_unexplored = n_unexplored/n_total
        p_explored = 1 - p_unexplored
        return p_explored
    
    def _get_obs_5x5_flat(self, padding_width=2):
        obs_x, obs_y = self.position
        #Add the padding width to the position, so we transition into 
        # the coordinate system of the padded grid
        r_pad, c_pad = self._get_padded_position(padding_width=padding_width)
        grid, padded = self._get_padded_worldmap(padding_width=padding_width)
        #Debug assertions. The position of the robot should never be inside a wall
        assert (grid[obs_x, obs_y] != self._symbol2idx['W'])
        assert (padded[r_pad, c_pad] != self._symbol2idx['W'])
        
        #Get the 5x5 patch centered around the slightly randomized position
        patch = padded[r_pad-padding_width:r_pad+padding_width+1,c_pad-padding_width:c_pad+padding_width+1]
        # Debug-Ausgabe:
        #print(f"[ENV] Grid:\n{grid}")
        #print(f"[ENV] Padded Grid:\n{padded}")
        #print(f"[ENV] Observation at {(r_pad,c_pad)} (5x5):\n{patch}")
        #print(f"[ENV] Index range: ({self._get_padded_index(r_pad, padding_width)},{self._get_padded_index(c_pad, padding_width)})")
        flat_patch = list(patch.flatten().astype(np.float32))
        flat_patch.append(obs_x)
        flat_patch.append(obs_y)
        
        # Debug-Ausgabe:
        #print(f"[ENV] Observation (flattened 3x3): {flat_patch}")
        return np.array(flat_patch)

    def _get_obs(self, padding_width=2):
        obs = self._get_obs_revealed_map(padding_width)
        #obs = self._get_obs_5x5_llm(padding_width)
        #obs = self._get_obs_5x5_flat(padding_width)
        return obs
    
    def reset(self, row=None, col=None):
        if row is None or col is None:
            self.position = (self.start_row, self.start_col)
        else:
            self.position = (row, col)
        self.step_count = 0
        self.world_seen = np.reshape(["?" for row in self.world for char in row], (len(self.world), len(self.world[0]))).tolist()
        
        obs = self._get_obs()
        # print(f"[ENV] Reset: Position={self.position}")
        return obs

    def step(self, action):
        self.step_count += 1
        r, c = self.position

        if action == 0:      # Süden
            nr, nc = r + 1, c
        elif action == 1:    # Osten
            nr, nc = r, c + 1
        elif action == 2:    # Norden
            nr, nc = r - 1, c
        elif action == 3:    # Westen
            nr, nc = r, c - 1
        elif action == -1:    # No movement
            nr, nc = r, c         
        else:
            raise ValueError(f"Ungültige Aktion: {action}")

        nr = np.clip(nr, 0, self.n_rows - 1)
        nc = np.clip(nc, 0, self.n_cols - 1)
        cell = self.world[nr][nc]

        # Stochastische Mauern (Ziffern 1–9)
        if cell.isdigit():
            prob = int(cell) / 10.0
            if np.random.rand() < prob:
                nr, nc = r, c
                cell = self.world[nr][nc]

        done = False
        if cell == 'W':
            reward = self.wall_reward
            nr, nc = r, c
        elif cell == '-':
            reward = self.minus_reward
        elif cell == ' ':
            reward = self.empty_reward
        elif cell == '+':
            reward = self.plus_reward
        elif cell == 'g':
            reward = self.goal_reward
            if not(self.exploration_mode):
                done = True
        else:
            reward = self.empty_reward

        #This is very bad design, but LLMs are assholes sometimes
        if action == -1:
            reward = self.no_action_reward
        
        if self.step_count >= self.max_steps:
            done = True

        #If all of the map is explored you are also done
        explored_all = bool(np.floor(self.get_reveal_percent()))
        done = done or explored_all
        
        self.position = (nr, nc)
        obs = self._get_obs()
        info = {'step': self.step_count}

        #print(f"[ENV] Step {self.step_count}: Pos={self.position}, Action={action}, Cell='{cell}', Reward={reward}, Done={done}")

        return obs, reward, done, info

    def render(self, mode='human', sleep=0.05, show_stats=True):
        import sys
        import os
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write('\033c')
        display = copy.deepcopy(self.world)
        r, c = self.position
        row = list(display[r])
        row[c] = '\033[1;31mO\033[0m'  # rotes O für Agent
        display[r] = ''.join(row)
        for line in display:
            print(line)
        if show_stats:
            print(f"Position: {self.position}")
            print(f"Step: {self.step_count}")
        time.sleep(sleep)
