import gym
from gym import spaces
import numpy as np
from gym.envs.classic_control import rendering
import pandas as pd
import json

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.viewer = rendering.Viewer(600,400)
        self.surround = ["leftlead","lead","rightlead","leftalongside","rightalongside","leftrear","rear","rightrear"]
        #Time step
        self.t = 0.04
        
        #Action threshold
        self.threshold = 5
        
        #Distance to farthest merging point(m)
        self.distance_to_max_merging_point_min = 0 
        self.distance_to_max_merging_point_max = 186.99
        
        #Longitudinal Velocity(m/s)
        self.vcc_max = 40
        self.vcc_min = 0
        
        #Longitudinal distance between mainroad vehicle and merging vehicle
        self.distance_to_merging_max = 186.99
        self.distance_to_merging_min = 0
        
        #Lateral distance between mainroad vehicle and merging vehicle
        self.y_merging_distance_min = 0
        self.y_merging_distance_max = 5
        
        #Longitudinal velocity difference of mainroad vehicle and merging vehicle
        self.vccdiff_merging_max = 20
        self.vccdiff_merging_min = -20
        
        #Lateral velocity of merging vehicle(m/s)
        self.merging_y_vcc_max = 10
        self.merging_y_vcc_min =0
        
        #Lead vehicle roadid
        self.lead_road_max =1
        self.lead_road_min =-1
        
        #Longitudinal distance between mainroad vehicle and lead vehicle
        self.distance_to_lead_max = 186.99
        self.distance_to_lead_min = 0
        
        #Longitudinal velocity difference of mainroad vehicle and lead vehicle
        self.vccdiff_lead_max = 20
        self.vccdiff_lead_min = -20
        #Action
        self.longitudinal_acc_max = 5
        # Example when using discrete actions:
        #self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        #self.observation_space = spaces.Box(low=0, high=255,shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        obs_low = np.array([self.distance_to_max_merging_point_min,
                            self.vcc_min,
                            self.distance_to_merging_min,
                            self.y_merging_distance_min,
                            self.vccdiff_merging_min,
                            self.merging_y_vcc_min,
                            self.lead_road_min,
                            self.distance_to_lead_min,
                            self.vccdiff_lead_min],dtype=np.float32)
        obs_high = np.array([self.distance_to_max_merging_point_max,
                            self.vcc_max,
                            self.distance_to_merging_max,
                            self.y_merging_distance_max,
                            self.vccdiff_merging_max,
                            self.merging_y_vcc_max,
                            self.lead_road_max,
                            self.distance_to_lead_max,
                            self.vccdiff_lead_max],dtype =np.float32)
        
        act_high = np.array([self.longitudinal_acc_max], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.state = None
        self.frame = None
        self.trajectory=None
        self.s=None
        self.a = None
    def seed(self):
        pass
    def step(self, action):
        self.state = np.array(self.state, dtype=np.float32)
        distance_to_max_merging_point,vcc,distance_to_merging,y_merging_distance,\
            vccdiff_merging,merging_y_vcc,lead_road,distance_to_lead,vccdiff_lead = self.state
        done = False
        self.frame = self.frame+1
        #get action
        ax = action
        #threshold
        ax = np.clip(ax,-self.threshold,self.threshold)
        ##compute delta_x
        lx = vcc*self.t+0.5*ax*self.t**2
        #get original next_state
        now_s = self.s[self.frame-1]
        next_s = self.s[self.frame]
        #compute_s[0]
        next_distance_to_max_merging_point = distance_to_max_merging_point-lx
        #compute_s[1]
        next_vcc = vcc+ax*self.t
        #compute_s[2]
        next_distance_to_megrging = next_s[2]+(now_s[0]-next_s[0])-lx
        #compute_s[3]
        next_y_merging_distance = next_s[3]
        #compute_s[4]
        next_vccdiff_merging = next_s[4]-(next_s[1]-now_s[1])+ax*self.t
        #compute_s[5]
        next_merging_y_vcc = next_s[5]
        #compute_s[6]
        next_lead_road = next_s[6]
        #compute_s[7]
        next_distance_to_lead = next_s[7]+(now_s[0]-next_s[0])-lx
        #compute_s[8]
        next_vccdiff_lead = next[8]-(next_s[1]-now_s[1])+ax*self.t
        

        #episode termination judgment
        done = bool(next_distance_to_max_merging_point<1 or next_distance_to_megrging<=3 \
            or (next_distance_to_lead<=3 and next_lead_road==1))
        self.state = [next_distance_to_max_merging_point,next_vcc,next_distance_to_megrging,next_y_merging_distance,next_vccdiff_merging,\
            next_merging_y_vcc,next_lead_road,next_vccdiff_lead]
        #get next_state
        obs = self.state
        reward = 0
        info={}
        self.state = [next_distance_to_max_merging_point,next_vcc,next_distance_to_megrging,next_y_merging_distance,next_vccdiff_merging,\
            next_merging_y_vcc,next_lead_road,next_vccdiff_lead]
        return np.array(self.state,dtype=np.float32),reward,done,info

    def reset(self,seed1,seed2):
        #Read the seed JSON trajectory file
        filepath = f"C:\\Users\\lenovo\\Desktop\\test_env\\merging_trajectory\\trajectory_t\\{seed1}\\{seed1}_{seed2}_trajectory.json"
        with open(filepath,"r") as f:
            self.trajectory = json.load(f)
        self.s = self.trajectory[::2]
        self.a = self.trajectory[1::2]
        self.state = self.trajectory[0]
        self.frame =0
        
        return np.array(self.state, dtype=np.float32) # reward, done, info can't be included

    def render(self, mode='human'):
        #There is no visualization requirement for this project
        self.viewer = rendering.Viewer()
        
    def close(self):
        pass
