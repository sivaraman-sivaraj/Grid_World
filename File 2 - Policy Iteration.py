import numpy as np
import matplotlib.pyplot as plt 
import gym,numpy as np,random,time,math,os,sys,pathlib
from gym.envs.registration import register
import Environment as E
from matplotlib import animation 

#########################################################################
#########################################################################
#########################################################################
#########################################################################
def get_base_kwargs():
    subgoal_location   = [(0, 9), (1, 3), (5, 2), (5, 7), (8, 4)]
    goal_location      = (9,9)
    action_stochasticity = [1.0, 0.0, 0.0, 0.0]
    non_terminal_reward  = -1
    subgoal_reward       = [13, 9, 7, 11, 5]
    terminal_reward      = 100

    base_kwargs =  {
            "subgoal_location": subgoal_location,
            "goal_location": goal_location,
            "action_stochasticity": action_stochasticity,
            "non_terminal_reward": non_terminal_reward,
            "subgoal_reward": subgoal_reward,
            "terminal_reward": terminal_reward,}
    
    return base_kwargs

base_kwargs = get_base_kwargs()
#########################################################################
#########################################################################
#########################################################################
#########################################################################

##################################################
##### Registering the Environment in Gym #########
##################################################
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
      if 'Gridworld-v0' in env:
          del gym.envs.registration.registry.env_specs[env]

register(id ='Gridworld-v0', entry_point='Environment: Env.load',kwargs={**base_kwargs}) 


#########################################################
#################### Policy Iteration ###################
#########################################################

def policy_iteration(env, gamma=0.99):
    
    # extracting no. of subgoals
    n = len(env.subgoal_reward)
    n2 = 2**n

    # Initial Values
    values = np.zeros((10, 10, n2))

    # Initial policy
    policy = np.empty((10, 10, n2), object)
    policy[:] = 'N' # Make all the policy values as 'N'


    N_s = n # number of subgoals
    
    def P_Evln(state,action):
      Ns_p_rs = env.get_transition_probabilites_and_reward(state,action)
      J = 0
      for j in range(len(Ns_p_rs)):
        p_    = Ns_p_rs[j][1]
        r_    = Ns_p_rs[j][2]
        v_nxt = values[Ns_p_rs[j][0][0][0]] [Ns_p_rs[j][0][0][1]] [E.binv2num(Ns_p_rs[j][0][1])]
        J     += p_ *( r_ + (gamma*v_nxt)) # value 
      return J 
    
    def P_Impt(state):
      V           = [ ]
      all_actions = env.perpendicular_order
      for i in range(len(all_actions)):
        v_temp = P_Evln(state,all_actions[i])
        V.append(v_temp) 
      policy   = all_actions[np.argmax(V)] 
      return policy
    
    
    ############################################
    ############################################
    cnt         = 0
    Flag        = True
    while Flag:
      cnt         += 1
      V_old        = np.copy(values)
      ####################################
      ######## Policy Evaluation #########
      ####################################
      for i1 in range(len(values)):
        for i2 in range(len(values[0])):
          for i3 in range(len(values[0][0])): #
              state    = [(i1,i2),E.num2binv(i3)]
              action   = policy[i1][i2][i3] 
              J_c      = P_Evln(state,action)
              values[i1][i2][i3] = J_c
      ##################################
      ###### Policy Improvement ########
      ################################## 
      for i1 in range(len(values)):
        for i2 in range(len(values[0])):
          for i3 in range(len(values[0][0])): #
              state    = [(i1,i2),E.num2binv(i3, N_s)]
              A_c      = P_Impt(state)
              policy[i1][i2][i3] = A_c
      if np.allclose(values, V_old, atol = 0.001):
        Flag = False 
        print("Yes..Converged...!")
      if cnt > 30: ## assertions for avoiding loop
        Flag = False
    
    extra_info = { }
    print(cnt)
    
    return {"Values": values, "Policy": policy}, extra_info

#########################################################
#################### Video Rendering ####################
#########################################################

def optimal_path(env, policy, start_state = [(0,0),[0]*5]):
    '''
    Code the optimal path function below, and return the optimal path matrix and
    as shown in the above example for a given policy.

    Hint: Use the env.step method to sample the rewards and states.
    '''
    
    grid = np.empty((10, 10), object)
    grid[:] = ' '
    path_reward = 0
    Flag = False
    cnt = 0
    state = env.reset()
    frames = []
    while not Flag:
        cnt +=1
        frames.append(env.render(mode="rgb_array"))
        frames.append(env.render(mode="rgb_array"))
        frames.append(env.render(mode="rgb_array"))
        time.sleep(0.6)
        i,j,k           = state[0][0],state[0][1],E.binv2num(state[1])
        p_temp          = policy[i][j][k]
        if grid[i][j] == ' ':
          grid[i][j]      = p_temp
        else:
          grid[i][j]         = str(grid[i][j][-1])+ p_temp
        NS,r,Flag            = env.step(state,p_temp)
        # NS,r,Flag            = env.step(state,env.action_space_sample()) 
        path_reward     += r
        state            = NS
        if Flag == True:
            frames.append(env.render(mode="rgb_array"))
            NS,r,Flag            = env.step(state,p_temp)
            path_reward     += r
            state            = NS
            
        if cnt > 25 :
            Flag = True
        
    env.close()
    E.Create_GIF(frames,filename='SS.gif')
    print("Optimal Path:")
    print(np.flipud(grid)) # flipud for accurate orientation
    print("Path Reward:")
    print(path_reward)

##################################################
#################### Execution ###################
##################################################
env             = gym.make('Gridworld-v0')
P_A,P_B         = policy_iteration(env, gamma=0.99)
policy_p        =  P_A["Policy"]
optimal_path(env, policy_p)
############################################
#################### End ###################
############################################












