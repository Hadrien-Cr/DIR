import sys
from Agent.agent import Agent_Tree_Search
from Agent.utils import round_dict
from Connect4.env_connect4 import Connect4Board
from Connect4.heuristic_connect4 import *
from time import time
from tqdm import tqdm



#test section
def matchup(env,agent1,agent2,initial_state=None,n_games=100,display=False,mix_side=True):
    '''

    matchup is a function that runs a match between agent1 and agent2 on the specified game environment.

    Args:
        - agent1,agent2: Agent_Tree_Search !: the agents that are competeting (see Agent.py for the requirements of the environment) 
        - initial_state=None: the initial state of each game if the initial state is not specified, the default initial_state env.initial_state() is used
        - n_games=10: The number of games to play
        - env: specified game environment. (see Agent.py for the requirements of the environment)
        - if  display=True, it displays all the states of all the games
        - if mix_side=True, both agent plays the same amount of games as player 1 and player 2
    Outputs
        - counts: a dictionnary of the results of the match in this format  {'Draws': n_draws/n_games,
                                                                            'Player 1 Wins': n_wins_1/n_games,
                                                                            'Player 2 Wins':n_wins_2/n_games,
                                                                            'mix_side':mix_side} (percentages)
        - avg_time_spent: The average time spent in seconds for each game {'Time_Agent1':0,'Time_Agent2':0}
        

    '''

    counts={'Draws':0,'Player 1 Wins':0,'Player 2 Wins':0} #keep tracks of the results of the games
    avg_time_spent={'Time_Agent1':0,'Time_Agent2':0} #time spent by each agent

    if initial_state==None:
        initial_state=env.initial_state()

    for _ in tqdm(range(n_games)):
        if mix_side:
            agent1.bit_player==n_games%2+1
            agent2.bit_player==(n_games+1)%2+1
        else:
            agent1.bit_player=1
            agent2.bit_player=2
        #initialize the game
        bit_player=1
        state=initial_state

        while True:
            #make the display
            if display:
                env.display_state(state)
                print('--------------------')

            #pick a move
            if bit_player==1:
                st=time()
                action=agent1.get_move(env,state)
                avg_time_spent['Time_Agent1']+=(time()-st)/n_games

            elif bit_player==2:
                st=time()
                action=agent2.get_move(env,state)
                avg_time_spent['Time_Agent2']+=(time()-st)/n_games


            #store the result of the game if the game ends
            if action==None or env.reward(state,action)!=0:  #stop the game
                if action==None: #draw
                    counts['Draws']+=1/n_games
                elif env.reward(state,action)==1: #1 wins
                    counts['Player 1 Wins']+=1/n_games
                elif env.reward(state,action)==-1: #2 wins
                    counts['Player 2 Wins']+=1/n_games
                break

            # update the state and bit_player
            state=env.get_next_state(state,action)
            bit_player=(1 if bit_player ==2 else 2)
    counts=round_dict(counts,3)
    counts['mix_side']=True
    avg_time_spent=round_dict(avg_time_spent,3)
    return()


env = Connect4Board(dim_col=5,dim_row=5)

agent1=Agent_Tree_Search(max_depth=5,method='alpha_beta_pruning',heuristic_reward=heuristic_reward_connect4,heuristic_sort=heuristic_sort_connect4,bit_player=1)
agent2=Agent_Tree_Search(method='monte-carlo',max_steps=120,repeat_sim=1,c=1,default_policy=random_policy_connect4,bit_player=2)

print(matchup(env,agent1,agent2,display=False,mix_side=True))


