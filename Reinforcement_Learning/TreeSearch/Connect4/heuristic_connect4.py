from random import choice

def heuristic_reward_connect4(env, state):
    l=env.get_available_actions(state,1)
    if len(l)==0:
        max_row=-1
    elif len(l)>0:
        max_row=sorted(l,key=(lambda action: action[0]))[0][0]
    counts=[0,0,0]

    for row in range(max_row+1):
        for col in range(env.dim_col):
            index = env.dim_col * row + col
            p=3**index
            bit_player=(state % (3 * p) - state % p) // p
            for step in [1,env.dim_col,env.dim_col+1,env.dim_col-1]:
                pos=-1
                to_break=False
                while to_break==False:     
                        index_neighbor = index+pos*step
                        p2 = 3 ** index_neighbor 
                        bit_neighbor = (state % (3 * p2) - state % p2) // p2
                        #print('index_neighbor',index_neighbor,'bit_neighbor',bit_neighbor,'pos',pos,'count',count)
                        if bit_neighbor == bit_player and index_neighbor>=0 and index_neighbor<=env.dim_col*env.dim_row and col+pos*((step+1)%env.dim_col-1)>= 0 and col+pos*((step+1)%env.dim_col-1)< env.dim_col:
                            counts[bit_player]+=(1-0.01*step)
                            pos+=(1 if pos>0 else -1)
                        else:
                            if pos<0:
                                pos=+1
                            else:
                                to_break=True

    return 10**(-5)*(counts[1]-counts[2])

def heuristic_sort_connect4(env, last_state,l,bit_player):
    l=sorted(l,key=(lambda action: abs(action[1]-env.dim_col/2)))
    return l

def random_policy_connect4(env,current_state,bit_player):
    return(choice(env.get_available_actions(current_state,bit_player)))

def policy_connect4_with_heuristic(env,current_state,bit_player):
    actions=env.get_available_actions(current_state,bit_player)
    return(max(actions, key = lambda action: heuristic_reward_connect4(env, env.get_next_state(current_state,action))))