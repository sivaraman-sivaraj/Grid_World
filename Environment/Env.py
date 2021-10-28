import numpy as np, os
from   gym import Env
from   gym.utils import seeding
from gym.envs.classic_control import rendering 
import matplotlib.pyplot as plt
from matplotlib import animation 
import  pyglet, time
###########################################################
############# Supplimentry Functions ######################
###########################################################
colors ={ "A"         : [0.94,0.94,0.94],     # Agent
         "N"         : [0, 0.74, 0.8],      # Normal Land
         "S"         : [0.74,0.74,0.74],
         "G"         : [0.75,0.75,0.0],       # Goal
         "SG"        : [0.68, 0, 0],         # Sub Goal 
         "A1"        : [0.44,0,0.44],           # Agent 1
         "Line-I"    : [0.0,0.0,0.0],
         "Line-O"    : [0.192,1,0.192],
         
         } 

def cell_corners(loc):
    x,y = loc[0]*40, loc[1]*40
    P = [(x,y)]
    P.append((x+40,y))
    P.append((x+40,y+40))
    P.append((x,y+40))
    return P 

def sub_goal_PM(SG):
    LP = []
    for i in range(len(SG)):
        LP.append(cell_corners((SG[i][1],SG[i][0]))) 
    return LP

    
def Edges1():
    P = []
    R = np.arange(0,401,40).tolist()
    for i in range(0,len(R)-2,2):
        P.append((R[i],0))
        P.append((R[i]+40,0))
        P.append((R[i]+40,400))
        P.append((R[i]+80,400))
        P.append((R[i]+80,0))
    return P

def Edges2():
    P = []
    R = np.arange(0,401,40).tolist()
    for i in range(0,len(R)-2,2):
        P.append((0,R[i]))
        P.append((0,R[i]+40))
        P.append((400,R[i]+40))
        P.append((400,R[i]+80))
        P.append((0,R[i]+80))
    return P

def binv2num(arr):
    '''
    Converts a binary vector (list) of any length to an integer.
    e.g. [0,1,0,1,1] -> 11.
    '''
    sum_ = 0
    
    for index, val in enumerate(reversed(arr)):
        sum_ += (val * 2**index)

    return int(sum_)

def num2binv(num, n=5):
    '''
    Converts an integer to a binary vector (list) of length n.
    e.g. 11 -> [0,1,0,1,1]
    '''
    binv = []
    B = np.binary_repr(num,width=n)
    binv = [int(x) for x in str(B)]

    return binv

###############################################
############### GIF Function ##################
###############################################
def Create_GIF(frames,filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 96.0, frames[0].shape[0] / 96.0), dpi=144)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        time.sleep(0.2)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=20)
    anim.save( os.path.join(os.getcwd(),filename), writer='imagemagick', fps=320) 
    
    
###########################################################
############# Environment Construction ####################
###########################################################

class load(Env): # NavigationRL-v0
     metadata = {'render.modes': ['human', 'rgb_array'],
                 'video.frames_per_second': 2 }
     
     def __init__(self,subgoal_location, goal_location, action_stochasticity,
                 non_terminal_reward, subgoal_reward, terminal_reward):
         
        '''
        Establishing the Environment.

        Parameters
        ----------
        (**kwargs)
        Returns
        -------
        environment 

        '''
        self.action_stochasticity   = action_stochasticity
        self.non_terminal_reward    = non_terminal_reward
        self.subgoal_reward         = subgoal_reward
        self.terminal_reward        = terminal_reward
        self.grid_size              = [10, 10]
        self.viewer                 =  None  
        self.st_x, self.st_y        = 0,0
        self.G                      = goal_location
        #############################
        ### Index of the actions ####
        ############################# 
        self.actions             = {'N': (1, 0),'E': (0,1),'S': (-1,0),'W': (0,-1)}
        self.perpendicular_order = ['N', 'E', 'S', 'W']
        
        l         = ['normal' for _ in range(self.grid_size[0]) ]
        self.grid = np.array([l for _ in range(self.grid_size[1]) ], dtype=object)

        # We number the subgoals in order of their appearance in the list 'subgoal_locations'
        for i, sg in enumerate(subgoal_location):
            self.grid[sg[0],sg[1]] = 'subgoal'+str(i+1)
        
        
        self.grid[goal_location[0], goal_location[1]] = 'goal'
        self.goal_location                            = goal_location
        
        self.subgoal_location                          = subgoal_location

        self.states_sanity_check()
        
     def states_sanity_check(self):
         pass
     
        
     def _out_of_grid(self, loc):
             
        # state = [(loc[0], loc[1]), [0, 0, ..., 0]]
        
        if loc[0] < 0 or loc[1] < 0:
            return True
        elif loc[0] > self.grid_size[0] - 1:
            return True
        elif loc[1] > self.grid_size[1] - 1:
            return True
        else:
            return False
    
        
     def _grid_state(self, loc):
         return self.grid[loc[0], loc[1]]        
            
     def get_transition_probabilites_and_reward(self, state, action):
        """ 
        Returns the probabiltity of all possible transitions for the given action in the form:
        A list of tuples of (next_state, probability, reward)
        Note that based on number of state and action there can be many different next states
        Unless the state is All the probabilities of next states should add up to 1
        """
        loc, binv = state[0], state[1]
        grid_state = self._grid_state(loc)
        
        if grid_state == 'goal':
            return [(state, 1.0, 0.0)]
        
        direction = self.actions.get(action, None)
        if direction is None:
            raise ValueError("Invalid action %s , please select among" % action, list(self.actions.keys()))

        nextstates_prob_rews = []

        ###########################################################
        ###########################################################
        def f(id): # will return the particular forward, backward,left and right direction
          Ref = self.perpendicular_order 
          F   = Ref[id] 
          B   = Ref[id - 2]
          L   = Ref[id - 1]
          R   = Ref[0] if id == 3 else Ref[id + 1] 
          return F,B,L,R

        #########################################
        D_id = 0  # Direction id
        for k in range(len(self.perpendicular_order)):
           if self.perpendicular_order[k] == action: # usage of action
             D_id   = k
        
        F,R,B,L   = f(D_id)
        
        #########################################
        P_OG = 0
        N_OG = 0 
        for item in self.actions:
          binv_copy     = binv.copy()
          (del_x,del_y) = self.actions[item]
          loc_new       = (loc[0]+del_x,loc[1]+del_y)               
          if item == F:
              p         = self.action_stochasticity[0]              
          elif item == R:
              p         = self.action_stochasticity[1]
          elif item == B:
              p         = self.action_stochasticity[2]
          elif item == L:
              p         = self.action_stochasticity[3]
          
          
          if self._out_of_grid(loc_new) == True:
            P_OG += p
            N_OG += 1 

          elif self._out_of_grid(loc_new) == False:                   
            
            new_grid_state  = self._grid_state(loc_new)                      
            if new_grid_state == 'normal':
              r  = self.non_terminal_reward
            elif new_grid_state == 'goal':
              r  = self.terminal_reward
            else:
              id = int(new_grid_state[-1])-1
              if binv_copy[id] != 1:
                r             = self.subgoal_reward[id]
                binv_copy[id] = 1
              else:
                r             = self.non_terminal_reward
            ns = [loc_new,binv_copy]
            nextstates_prob_rews.append((ns,p,r))
        
        if N_OG != 0:
            new_grid_state  = self._grid_state(loc)     
            binv_og       = binv.copy()
                 
            if new_grid_state == 'normal':
                r  = self.non_terminal_reward
            elif new_grid_state == 'goal':
                r  = self.terminal_reward
            else:
                id = int(new_grid_state[-1])-1
                if binv_og[id] != 1:
                  r             = self.subgoal_reward[id]
                  binv_og[id] = 1
                else:
                  r             = self.non_terminal_reward
            ns_og = [loc,binv_og]
            nextstates_prob_rews.append((ns_og,P_OG,r))
        return nextstates_prob_rews
    
     def step(self, state, action):
         self.npr = self.get_transition_probabilites_and_reward(state, action)
         self.probs = [t[1] for t in self.npr]
         self.sampled_idx = np.random.choice(range(len(self.npr)), p=self.probs)
         sampled_npr = self.npr[self.sampled_idx]
         self.c_state = sampled_npr[0]
         self.reward = sampled_npr[2]
         is_terminal = self.c_state[0] == tuple(self.goal_location)
         self.current_state = self.c_state
         return self.c_state, self.reward, is_terminal
     
     def reset(self):
        self.done                             = False
        self.current_state                    = [(0,0),[0]*len(self.subgoal_reward)]
        return self.current_state
    
    
     def action_space_sample(self):
        n = np.random.choice(self.perpendicular_order)
        return n
    
     def action_space(self):
        return self.perpendicular_order  
     
        
     def render(self,mode='human'):
         screen_width     = 400
         screen_height    = 400 
         self.loc         = self.current_state[0]
         if self.viewer is None:
             self.viewer = rendering.Viewer(screen_width, screen_height)
             ##########################################
             ############## Background ################
             ##########################################
             left_land = rendering.FilledPolygon([(0,0),(400,0),(400,400),(0,400)])
             left_land.set_color(colors["N"][0],colors["N"][1],colors["N"][2])
             self.viewer.add_geom(left_land)
             ##########################################
             ############## Goal ######################
             ##########################################
             self.goal = rendering.FilledPolygon(cell_corners(self.G))
             self.goal.set_color(colors["G"][0],colors["G"][1],colors["G"][2])
             self.viewer.add_geom(self.goal)
             
             ##########################################
             ############## Starting Point ############
             ##########################################
             self.st = rendering.FilledPolygon([(0,0),(40,0),(40,40),(0,40)])
             self.st.set_color(colors["S"][0],colors["S"][1],colors["S"][2])
             self.viewer.add_geom(self.st)
             ###################################################
             ############### Edge Rendering ####################
             ###################################################
             self.x_axis = rendering.make_polyline(Edges1())
             self.x_axis.set_color(colors["Line-I"][0],colors["Line-I"][1],colors["Line-I"][2])
             self.viewer.add_geom(self.x_axis)
             
             self.y_axis = rendering.make_polyline(Edges2())
             self.y_axis.set_color(colors["Line-I"][0],colors["Line-I"][1],colors["Line-I"][2])
             self.viewer.add_geom(self.y_axis)
             
             self.e_axis = rendering.make_polyline([(0,0),(400,0),(400,400),(0,400),(0,0)])
             self.e_axis.set_color(colors["Line-I"][0],colors["Line-I"][1],colors["Line-I"][2])
             self.viewer.add_geom(self.e_axis)
             
             self.e_axis = rendering.make_polyline([(0,0),(399,1),(399,399),(1,399),(0,0)])
             self.e_axis.set_color(colors["Line-I"][0],colors["Line-I"][1],colors["Line-I"][2])
             self.viewer.add_geom(self.e_axis)
             ##########################################
             ############### Sub Goals ################
             ##########################################
             self.SG   = sub_goal_PM(self.subgoal_location)
             for i in range(len(self.SG)):
                 sub_goal = rendering.FilledPolygon(self.SG[i])
                 sub_goal.set_color(colors["SG"][0],colors["SG"][1],colors["SG"][2])
                 self.viewer.add_geom(sub_goal)
             ##########################################
             ############## Agent #####################
             ##########################################
             agent = rendering.FilledPolygon(cell_corners((0,0)))
             agent.set_color(colors["A"][0],colors["A"][1],colors["A"][2])
             self.atrans = rendering.Transform() 
             agent.add_attr(self.atrans)
             self.viewer.add_geom(agent)
             
             self.axle = rendering.FilledPolygon([(10,10),(10,30),(30,30),(30,10)])
             self.axle.add_attr(self.atrans)
             self.axle.set_color(colors["A1"][0],colors["A1"][1],colors["A1"][2])
             self.viewer.add_geom(self.axle)
         agent_x,agent_y = self.loc[1]*40,self.loc[0]*40
         self.atrans.set_translation(agent_x,agent_y)
         return self.viewer.render(return_rgb_array=mode == 'rgb_array')
     
     
     
     def close(self):
         if self.viewer:
             self.viewer.close()
             self.viewer = None
             
             
                 
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             