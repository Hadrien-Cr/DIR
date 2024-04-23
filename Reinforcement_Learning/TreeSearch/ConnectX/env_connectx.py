'''

An instance of the class Board represent an environment of the game of Connect4.
The first player to play has the bit_player 1 and the second player to play has the bit_player 2.
The first player to play places the symbol 'X' and the second player to places the symbol 'O'

The class ConnectXBoard have 6 key methods:

    - init(self,dim_row,dim_col,in_a_row): initialize and choose the size of the board and the winning cond

    - display(self,state): display the state

    - initial_state(self): return the initial state of the game

    - get_available_actions(self,state,bit_player): give all the possible actions 

    - get_next_state(self,state,action): computes the next state associated to the action and the current state

    - eval(self,state,action) computes the reward associated to the action (+1 if the player 1 has just won, -1 if the player 1 has just won, else 0)


'''


class ConnectXBoard:
    def __init__(self,dim_row=6,dim_col=7,in_a_row=4):
        self.dim_row=dim_row
        self.dim_col=dim_col
        self.in_a_row=in_a_row

    def initial_state(self):
        return(0)
    
    def display_state(self,state):
        mapping=['_','X','O']
        board=[['_' for _ in range(self.dim_col)] for _ in range(self.dim_row)]

        for row in range(self.dim_row):
            for col in range(self.dim_col):
                index=self.dim_col*row+col
                p=3**index
                val = (state % (3 * p) - state % p) // p
                board[row][col]=mapping[val]

        for row in range(self.dim_row-1,-1,-1):
            print(board[row])

    def reward(self, last_state, action):
            row, col, bit_player = action
            index = self.dim_col * row + col
            for step in [1,self.dim_col,self.dim_col+1,self.dim_col-1]:
                count=0
                pos=-1
                to_break=False
                while count<self.in_a_row-1 and to_break==False:     
                        index_neighbor = index+pos*step
                        p2 = 3 ** index_neighbor 
                        bit_neighbor = (last_state % (3 * p2) - last_state % p2) // p2
                        if bit_neighbor == bit_player and index_neighbor>=0 and index_neighbor<=self.dim_col*self.dim_row and col+pos*((step+1)%self.dim_col-1)>= 0 and col+pos*((step+1)%self.dim_col-1)< self.dim_col:
                            count+=1
                            pos+=(1 if pos>0 else -1)
                        else:
                            if pos<0:
                                pos=+1
                            else:
                                to_break=True
                if count==self.in_a_row-1:
                    return((1 if bit_player==1 else -1))
            return 0
    
    def get_next_state(self,state,action):
        if action==None:
            return(state)
        else:
            row,col,bit_player=action
            index=self.dim_col*row+col
            next_state=state+bit_player*3**index
            return(next_state)
        
    def update_state(self,action):
        self.state=self.get_next_state(self.state,action)

    def terminate(self,square,smallest_moves):
        if square%self.dim_col == 0:
            for move in smallest_moves:
                if move>=square//self.dim_col:
                    return(False)
            return(True)
        else:
            return(False)

    def get_available_actions(self,state,bit_player):
        smallest_moves=[0 for col in range(self.dim_col)]
        x=state
        square=0
        
        while x>0 and not self.terminate(square,smallest_moves):
            row,col=square//self.dim_col,square%self.dim_col
            q,r=x//3,x%3
            if r!=0:
                smallest_moves[col]=row+1
            square+=1
            x=q
        l=[]
        for col in range(self.dim_col):
            if smallest_moves[col]<self.dim_row:
                l.append((smallest_moves[col],col,bit_player))  
        return(l)


