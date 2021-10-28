import numpy as np
import gym,numpy as np,random,time,math,os,sys,pathlib
from gym.envs.registration import register
import Environment as E
from matplotlib import animation 
import Environment as E

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


##################################################
#################### Value Iteration #############
##################################################

def value_iteration(env, gamma=0.99):
    '''
    Code your value iteration algorithms here, the function should return a np 
    array of dimensions (10, 10, 2^(no. of sub-goals)). Also return the final
    policy grid corresponding to the final values.
    '''
    
    # extracting no. of subgoals
    n = len(env.subgoal_reward)
    n2 = 2**n

    # Initial Values
    values = np.zeros((10, 10, n2))

    # Initial policy
    policy = np.empty((10, 10, n2), object)
    policy[:] = 'N' # Make all the policy values as 'N'


    ##################################################
    ############### Begin code here ##################
    ##################################################
    n1=10

    def P_Evln(state,act):
      Ns_p_rs = env.get_transition_probabilites_and_reward(state,act)
      J = 0
      for j in range(len(Ns_p_rs)):
        p_    = Ns_p_rs[j][1]
        r_    = Ns_p_rs[j][2]
        v_nxt = values[Ns_p_rs[j][0][0][0]] [Ns_p_rs[j][0][0][1]] [E.binv2num(Ns_p_rs[j][0][1])]
        J     += p_ *( r_ + (gamma*v_nxt)) # value 
      return J 

    def V_max(state):
      V           = [ ]
      all_actions = env.perpendicular_order
      for i in range(len(all_actions)):
        v_temp  = P_Evln(state,all_actions[i])
        V.append(v_temp) 
      V_return = max(V)
      policy   = all_actions[np.argmax(V)] 
      return V_return,policy 

    ####################################
    ######### Value Iteration ##########
    ####################################
    PL,cnt      = [],0
    
    Flag  = True 
    while Flag:
      cnt      += 1
      V_old    = np.copy(values)
      for i1 in range(n1):
        for i2 in range(n1): 
          for i3 in range(n2): #
            state    = [(i1,i2),E.num2binv(i3)]
            v_c, p_c = V_max(state)                # Standard Value Iteration
            values[i1][i2][i3] = v_c 
            policy[i1][i2][i3] = p_c               # Policy Update 
            if i1 == i2 == 9:
              policy[i1][i2][i3] = 'E'
      if np.allclose(values, V_old, atol = 0.01):
        Flag = False 
        print("yes.. converged")
      if cnt > 150:
        Flag = False

      V_old = np.copy(values)
      PL.append(np.mean(V_old))

    
    ######### ###########
    # Put your extra information needed for plots etc in this dictionary
    extra_info = {}
    print(np.mean(values))
    print("Iteration : ", cnt)
    #####################
    ##### End code ######
    #####################


    # Do not change the number of output values
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
        time.sleep(0.9)
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

############################################
#################### Execution #############
############################################
env             = gym.make('Gridworld-v0')
V_A,V_B         = value_iteration(env, gamma=0.99)
policy_v        =  V_A["Policy"]
optimal_path(env, policy_v)
############################################
#################### End ###################
############################################












