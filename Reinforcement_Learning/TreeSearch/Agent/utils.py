import random
from math import log
import networkx as nx
import matplotlib.pyplot as plt
def round_dict(d,p):
    for key,value in d.items():
        d[key]=round(value,p)
    return(d)

class Node:

    '''
    Class created for the nodes of a Monte Carlo Tree Seach
    '''

    def __init__(self, current_state, prev_node,bit_player):
        self.bit_player = bit_player # player that makes a move which leads to one of the child nodes
        self.state = current_state
        self.prev_node = prev_node
        self.children = dict() # {action: Node}

        self.is_leaf = True

        self.wins = 0. # number of games won by player 1
        self.visits = 0 # number of games played where node was traversed

    def eval_ucb(self,c,t,bit_player): #UCB evaluation
        x=(1 if bit_player==1 else -1)
        return (x*self.wins / self.visits + c*(log(t)/self.visits)**0.5 if self.visits > 0 else float("inf"))

    def add_child(self, bit_next_player, next_state, action):
        self.children[action] = Node(next_state, prev_node = self, bit_player=bit_next_player)

    def choose_best_action_ucb(self,c):
        return max(self.children, key = lambda action: self.children[action].eval_ucb(c,t=self.visits,bit_player=self.bit_player))
 
    def choose_random_action(self):
        return random.sample(list(self.children), 1)[0]

def selection(node,c): #selects the path fund by the "tree policy"
    path = [node]
    while path[-1].is_leaf is False: # stop when we are out of the "tree policy"
        node=path[-1]
        action = node.choose_best_action_ucb(c)
        path.append(node.children[action])
    return path

def expansion(env,path): #add an additional node is selected to the end of the path
    node=path[-1]
    actions=env.get_available_actions(node.state,node.bit_player)
    if node.visits>0 and len(actions)>0:
        node.is_leaf= False
        for action in actions:
            next_bit_player=(2 if node.bit_player==1 else 1)
            node.add_child(bit_next_player=next_bit_player,action=action,next_state=env.get_next_state(node.state,action))
        action = node.choose_random_action()
        path.append(node.children[action])

def simulate_full_game(env,current_state,default_policy,bit_player):
        actions=env.get_available_actions(current_state,bit_player)
        if len(actions)==0: #draw
            return(0)
         
        else:
            action=default_policy(env,current_state,bit_player)
            next_state=env.get_next_state(current_state,action)
            if env.reward(current_state,action)!=0:
                return(env.reward(current_state,action))
            else:
                bit_player=(2 if bit_player==1 else 1)
                return(simulate_full_game(env,next_state,default_policy,bit_player))

def repeat_simulate_full_game(env,current_state,default_policy,bit_player,repeat_sim=1):
    result=0
    for _ in range(repeat_sim):
        result+=simulate_full_game(env,current_state,default_policy,bit_player)
    return(result)

def backpropagation(result,path,repeat_sim=1):
    for node in path:
        node.wins+=result
        node.visits+=repeat_sim
        
def display_tree(root,branching_rate):
    """
    Function to visualize the game tree created by a call of agent.monte_carlo_sarch.
    Args:
        root (Node): The root node of the tree.
    """
    G = nx.DiGraph()
    x_min,x_max=-1000,+1000

    pos={}
    node_labels={}

    def add_to_graph(node, parent=None, action=None,x=0,n=0):
        # Add the node to the graph
        node_labels[node.state] = f"{node.state}\n(Visits: {node.visits}) \n (Wins: {node.wins})"

        G.add_node(node.state)
        pos[node.state]=(x,n)


        # If there's a parent, add an edge from the parent to the node
        if parent is not None and action is not None:
            G.add_edge(parent.state, node.state, label=action)
        
        # Recursively add children to the graph
        u,p=x_min,0
        k=len(node.children.items())
        for action, child in node.children.items():
            add_to_graph(child, node, action,x+(u+p*(x_max-x_min)/k)*(branching_rate+1)**(-n-1),n+1)
            p+=1
    
    # Start the recursive function from the root node
    add_to_graph(root)
    
    # Draw the graph with the specified layout
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue', font_size=6, font_color='black')
    
    # Draw edge labels (actions)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    # Show the graph
    plt.show()

