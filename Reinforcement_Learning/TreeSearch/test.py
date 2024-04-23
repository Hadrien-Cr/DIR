import sys
from Agent.agent import Agent_Tree_Search
from Agent.utils import *
from Connect4.env_connect4 import Connect4Board
from ConnectX.env_connectx import ConnectXBoard
from ConnectX.heuristic_connectx import *
from time import time
from tqdm import tqdm



#test section
def matchup_agent_vs_agent(env,agent1,agent2,initial_state=None,n_games=100,display=False,mix_side=True):
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

    for game in tqdm(range(n_games)):
        #initialize the game
        if mix_side:
            agent1.bit_player=game%2+1
            agent2.bit_player=(game+1)%2+1
        else:
            agent1.bit_player=1
            agent2.bit_player=2
        
        bit_player=1
        state=initial_state

        while True:
            #make the display
            if display:
                env.display_state(state)
                print('--------------------')

            #pick a move
            if bit_player==agent1.bit_player:
                st=time()
                action=agent1.get_move(env,state)
                avg_time_spent['Time_Agent1']+=(time()-st)/n_games

            elif bit_player==agent2.bit_player:
                st=time()
                action=agent2.get_move(env,state)
                avg_time_spent['Time_Agent2']+=(time()-st)/n_games


            #store the result of the game if the game ends
            if action==None or env.reward(state,action)!=0:  #stop the game
                if action==None: #draw
                    counts['Draws']+=1/n_games
                elif (env.reward(state,action)==1 and agent1.bit_player==1) or (env.reward(state,action)==-1 and agent1.bit_player==2) : #1 wins
                    counts['Player 1 Wins']+=1/n_games
                elif (env.reward(state,action)==1 and agent2.bit_player==1) or (env.reward(state,action)==-1 and agent2.bit_player==2) : #2 wins
                    counts['Player 2 Wins']+=1/n_games
                break

            # update the state and bit_player
            state=env.get_next_state(state,action)
            bit_player=(1 if bit_player ==2 else 2)

    counts=round_dict(counts,3)
    counts['mix_side']=True
    avg_time_spent=round_dict(avg_time_spent,3)
    return(counts,avg_time_spent)

def matchup_agent_vs_human(env,agent,initial_state=None,n_games=1,display=True,mix_side=True):
    '''

    matchup is a function that runs a match between agent1 and agent2 on the specified game environment.

    Args:
        - agent1: Agent_Tree_Search !: the agents that are competeting (see Agent.py for the requirements of the environment) 
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

    counts={'Draws':0,'Player Agent Wins':0,'Player Human Wins':0} #keep tracks of the results of the games
    avg_time_spent={'Time_Agent':0} #time spent by agent

    if initial_state==None:
        initial_state=env.initial_state()

    for game in tqdm(range(n_games)):
        #initialize the game
        if mix_side:
            agent.bit_player=game%2+1
            human_bit_player=(game+1)%2+1
        else:
            agent.bit_player=1
            human_bit_player=2
        
        state=initial_state
        bit_player=1

        while True:
            #make the display
            if display:
                env.display_state(state)
                print('--------------------')

            #pick a move
            if bit_player==agent.bit_player:
                print('Waiting for Agent to play')
                st=time()
                action=agent.get_move(env,state)
                avg_time_spent['Time_Agent']+=(time()-st)/n_games

            elif bit_player==human_bit_player:
                actions=env.get_available_actions(state,bit_player)
                print([((f'Action {i+1}'),action) for i,action in enumerate(actions)])
                number_selected=None
                test=0
                while not number_selected in [i+1 for i in range(len(actions))]:
                    print(f'Choose an action (write the correspong number) (Tries:{test})')
                    number_selected=int(input())
                    test+=1
                action=actions[number_selected-1]
                


            #store the result of the game if the game ends
            if action==None or env.reward(state,action)!=0:  #stop the game
                if action==None: #draw
                    print('Game is a draw')
                    counts['Draws']+=1/n_games
                elif (env.reward(state,action)==1 and agent.bit_player==1) or (env.reward(state,action)==-1 and agent.bit_player==2) : #1 wins
                    print('Game won by Agent')
                    counts['Player Agent Wins']+=1/n_games
                elif (env.reward(state,action)==1 and human_bit_player==1) or (env.reward(state,action)==-1 and human_bit_player==2) : #2 wins
                    print('Game won by Human')
                    counts['Player Human Wins']+=1/n_games
                break

            # update the state and bit_player
            state=env.get_next_state(state,action)
            bit_player=(1 if bit_player ==2 else 2)


    counts=round_dict(counts,3)
    counts['mix_side']=True
    avg_time_spent=round_dict(avg_time_spent,3)
    return(counts,avg_time_spent)



env = ConnectXBoard(dim_col=7,dim_row=6,in_a_row=4)

#agent1=Agent_Tree_Search(max_depth=9,method='alpha_beta_pruning',heuristic_reward=heuristic_reward_connectx,heuristic_sort=heuristic_sort_connectx,bit_player=1)
agent2=Agent_Tree_Search(method='monte-carlo',max_steps=1000,repeat_sim=1,c=0.1,default_policy=random_policy_connectx,bit_player=2)

#print(matchup_agent_vs_agent(env,agent1=agent1,agent2=agent2,n_games=10,mix_side=True))

print(matchup_agent_vs_human(env,agent=agent2,display=True,n_games=1,mix_side=True))


