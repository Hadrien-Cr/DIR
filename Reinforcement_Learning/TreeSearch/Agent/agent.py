from math import inf
from random import choice
import sys
sys.path.append( 'Reinforcement_Learning\TreeSearch\Agent' )
from utils import *

'''
DOCS

This is the documentation of the Agent_Tree_Search 

    - __init__(self, bit_player=None,method=None,
                heuristic_reward=None,heuristic_sort=None,default_policy=None,
                max_depth=5,max_steps=100,c=1,repeat_sim=10):

        method is either 'minimax','alpha_beta_pruning' or 'monte-carlo'
        Args:
            - bit_player: The first player to play has the bit_player 1 and the second player to play has the bit_player 2.

            - method: the method of the tree Search
                - If the method is 'minimax', an heuristic_reward have to be provided (function from the states of the env to float)

                - If the method is 'alpha_beta_pruning', an heuristic_reward(env,state)  (function from the states of the env to float)
                and an heuristic_sort(env,state,actions,bit_player) function have to be provided (function that return actions sorted by preference)

                - If the method is 'monte-carlo', a default_policy(env,state) have to be provided  (function from the states of the env to float)
                
            - (non 'monte-carlo' only) max_depth(optional,default=5): the maximum_depth of the minimax or alpha_beta_pruning search
            
            - ('monte-carlo' only) max_steps(optional,default=100): 
            - ('monte-carlo' only) c (optional,default=1): the parameter for the UCB evaluation of the nodes. A value between 1 and 5 is recommended.
            - ('monte-carlo' only) repeat_sim (optional,default=1): the number of times a simulation of a game should be repeated
                    
    - get_move(self, env, current_state) that proposes the best move found for the current state

    
For an agent to work properly, the environment 'env' has to have the following methods

    - get_available_actions(self,state,bit_player): give all the possible actions 

    - get_next_state(self,state,action): computes the next state associated to the action and the current state

    - eval(self,state,action) computes the reward associated to the action (+1 if the player 1 has just won, -1 if the player 1 has just won, else 0)

'''


class Agent_Tree_Search:
    def __init__(self, bit_player=None,method=None,
                heuristic_reward=None,heuristic_sort=None,default_policy=None,
                max_depth=5,max_steps=100,c=1,repeat_sim=10):
        self.max_depth = max_depth
        self.bit_player = bit_player
        self.method=method
        assert method in ['minimax','alpha_beta_pruning','monte-carlo'], 'invalid method name'

        if method in ['minimax','alpha_beta_pruning']: 
            assert heuristic_reward!= None,'missing "heuristic_reward" argument'
            self.heuristic_reward=heuristic_reward

        if method in ['alpha_beta_pruning']: 
            assert heuristic_sort!= None, 'missing "heuristic_sort" argument'
            self.heuristic_sort=heuristic_sort
            
        elif method in ['monte-carlo']: 
            assert default_policy!= None, 'missing "default_policy" argument'
            self.default_policy=default_policy

        self.max_steps=max_steps
        self.c=c
        self.repeat_sim=repeat_sim
    
    
    def minimax_search(self,env,heuristic_reward,heuristic_sort,current_state, depth=None, bit_player=None):

        actions=heuristic_sort(env,current_state,env.get_available_actions(current_state,bit_player),bit_player)
        #leaf
        if depth==0 or len(actions)==0:
            return(None,heuristic_reward(env,current_state))
        
        #not leaf, maximizing player
        if bit_player==1:
            max_eval = -inf
            best_action=actions[0]
            for action in actions:
                #if a winning move has been found
                if env.reward(current_state,action) == 1:
                    return(action,1)
                else:
                #else if the move is not winning
                    next_state=env.get_next_state(current_state,action)
                    eval=self.minimax_search(env,heuristic_reward,heuristic_sort,next_state, depth=depth-1, bit_player=2)[1]
                    if eval>max_eval:
                        best_action=action
                        max_eval =  eval
            return(best_action,max_eval)
        
        #not leaf, minimizing player
        if bit_player==2:
            min_eval = +inf
            best_action=actions[0]
            for action in actions:
                #if a winning move has been found
                if env.reward(current_state,action) == -1:
                    return(action,-1)
                #else if the move is not winning
                else:
                    next_state=env.get_next_state(current_state,action)
                    eval=self.minimax_search(env,heuristic_reward,heuristic_sort,next_state, depth=depth-1, bit_player=1)[1]

                    if eval<min_eval:
                        best_action=action
                        min_eval =  eval

            return(best_action,min_eval)
        

    def alpha_beta_pruning_search(self,env,heuristic_reward,heuristic_sort,current_state, alpha, beta, depth=None, bit_player=None):

        actions=heuristic_sort(env,current_state,env.get_available_actions(current_state,bit_player),bit_player)

        #leaf
        if depth==0 or len(actions)==0:
            return(None,heuristic_reward(env,current_state))
        
        
        #not leaf, maximizing player
        if bit_player==1:
            max_eval = -inf
            best_action=actions[0]
            for action in actions:
                #if a winning move has been found
                if env.reward(current_state,action) == 1:
                    return(action,1)
                else:
                #else if the move is not winning
                    next_state=env.get_next_state(current_state,action)
                    eval=self.alpha_beta_pruning_search(env,heuristic_reward,heuristic_sort,next_state, alpha, beta, depth=depth-1, bit_player=2)[1]
                    if eval>max_eval:
                        best_action=action
                        max_eval =  eval
                    if beta<=max_eval:
                        break
                    alpha = max(alpha, max_eval)
            return(best_action,max_eval)
        
        #not leaf, minimizing player
        if bit_player==2:
            min_eval = +inf
            best_action=actions[0]
            for action in actions:
                #if a winning move has been found
                if env.reward(current_state,action) == -1:
                    return(action,-1)
                #else if the move is not winning
                else:
                    next_state=env.get_next_state(current_state,action)
                    eval=self.alpha_beta_pruning_search(env,heuristic_reward,heuristic_sort, next_state, alpha, beta, depth=depth-1, bit_player=1)[1]
                    
                    if eval<min_eval:
                        best_action=action
                        min_eval = eval
                    if min_eval<=alpha:
                        break
                    beta = min(beta, min_eval)

            return(best_action,min_eval)


    def monte_carlo_search(self,env,root_state,default_policy,bit_player=None,max_steps=100,c=1,repeat_sim=1):

        root=Node(root_state,None,bit_player)
        step=0
        t=0
        path=selection(env,root,c)
        expansion(env,path)

        while step<max_steps:
            step+=1
            path=selection(env,root,c)
            expansion(env,path)
            last_node=path[-1]
            if  last_node.is_terminal[0]==True:
                result=last_node.is_terminal[1]
            else:  
                result=repeat_simulate_full_game(env,last_node.state,default_policy,last_node.bit_player,repeat_sim)

            backpropagation(result,path,repeat_sim)    



            '''
            print('___________________')
            print('path n ',k)
            for node in path:
                print('------------')
                env.display_state(node.state)
                print('visits: ',node.visits)
                print('eval: ',node.wins/node.visits)
            '''
        results=[(action,root.children[action].eval_ucb(c=0,t=1,bit_player=bit_player)) for action in env.get_available_actions(root_state,bit_player)]
        best_action,max_eval=max(results,key=(lambda x: x[1]))
        #display_tree(root,branching_rate=len(root.children))

        return(best_action,max_eval)


    def get_move(self, env, current_state):
        if len(env.get_available_actions(current_state,self.bit_player))>0:
            if self.method== 'minimax':
                best_action,minimax_score = self.minimax_search(env, self.heuristic_reward,self.heuristic_sort,current_state=current_state, depth=self.max_depth, bit_player=self.bit_player)
                return best_action,minimax_score
            elif self.method=='alpha_beta_pruning':
                best_action,score = self.alpha_beta_pruning_search(env, self.heuristic_reward,self.heuristic_sort,current_state=current_state, depth=self.max_depth, alpha=-inf, beta=inf, bit_player= self.bit_player)
                return(best_action)
            elif self.method=='monte-carlo':
                best_action,score=self.monte_carlo_search(env,current_state,self.default_policy,self.bit_player,self.max_steps,self.c,self.repeat_sim)
                return(best_action)
        else:
            return(None)
        

