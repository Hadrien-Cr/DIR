from numpy import random,argmax
import numpy as np
from statistics import mean
from math import *
from tqdm import tqdm

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class kArmedBanditProblem:
    # a k-armed bandit problem where all reward distribution follow a different normal distributions
    # reward(a) follows the normal distibutions of mean means[a] and variance stds[a]**2
    def __init__(self,k,means,stds):
        self.k=k
        self.means=means
        self.stds=stds
        assert len(means)==k
        assert len(stds)==k

    def draw(self,a):
        #draw a reward with action a
        return(random.normal(loc=self.means[a], scale=self.stds[a]))

class AgentGreedy:
    def __init__(self,epsilon,k,q1=0,step_rule='standard',alpha=None):
        self.epsilon=epsilon
        self.estimates=[q1 for a in range(k)] #estimates
        self.n=[0 for a in range(k)] #number of times moves played
        self.k=k
        self.step_rule=step_rule
        if step_rule=='constant':
            self.alpha=alpha
        self.history=[]

    def greedy_move(self):
        return(argmax(self.estimates))
    
    def exploratory_move(self):
        return(random.choice(self.k))
    
    def update(self,a,reward):
        self.n[a]+=1
        if self.step_rule=='constant':
            x=self.alpha*(reward-self.estimates[a])
            self.estimates[a]+=x
        elif self.step_rule=='standard':
            x=(1/self.n[a])*(reward-self.estimates[a])
            self.estimates[a]+=x
        else:
            print('step rule not implemented')

    def play(self,problem):
        if random.random()>self.epsilon:
            a=self.greedy_move()
        else:
            a=self.exploratory_move()
        reward=problem.draw(a)
        self.update(a,reward)
        self.history.append(reward)

class AgentUCB:
    def __init__(self,c,k,q1=0,step_rule='standard',alpha=None):
        self.c=c
        self.estimates=[q1 for a in range(k)] #estimates
        self.n=[0 for a in range(k)] #number of times moves played
        self.k=k
        self.step_rule=step_rule
        self.t=0
        if step_rule=='constant':
            self.alpha=alpha
        self.history=[] #history of reward
    def update(self,a,reward):
        self.n[a]+=1
        
        if self.step_rule=='constant':
            x=self.alpha*(reward-self.estimates[a])
            self.estimates[a]+=x
        elif self.step_rule=='standard':
            x=(1/self.n[a])*(reward-self.estimates[a])
            self.estimates[a]+=x
        else:
            print('step rule not implemented')


    def play(self,problem):
        self.t+=1
        values_ucb=[self.estimates[a] + self.c*(log(self.t)/self.n[a])**0.5 if self.n[a]>0 else inf for a in range(self.k)]
        a=argmax(values_ucb)
        reward=problem.draw(a)
        self.update(a,reward)
        self.history.append(reward)

class AgentGrad:
    def __init__(self,k,h1=0,r1=0,alpha=0.1,step_rule='standard',step_size=None):
        self.relative_preferences=[h1 for a in range(k)] #estimates
        self.baseline=r1
        self.k=k
        self.alpha=alpha
        self.step_rule=step_rule
        self.t=0
        if step_rule=='constant':
            self.step_size=step_size
        self.history=[] #history of reward


        
    def update(self,a,reward):
        #update the preferences
        softmaxs=softmax(self.relative_preferences)
        self.relative_preferences[a]+=self.alpha*(reward-self.baseline)*(1-softmaxs[a])
        for b in range(self.k):
            if b!=a:
                 self.relative_preferences[a]+=self.alpha*(reward-self.baseline)*(softmaxs[b])
        #update the baseline
        self.t+=1
        if self.step_rule=='constant':
            x=self.alpha*(reward-self.baseline)
            self.baseline+=x
        elif self.step_rule=='standard':
            x=(1/self.t)*(reward-self.baseline)
            self.baseline+=x
        else:
            print('step rule not implemented')

    def pick_move(self):
        probabilities = softmax(self.relative_preferences)
        a= np.random.choice(len(probabilities), p=probabilities)
        return a

    def play(self,problem):
        a=self.pick_move()
        reward=problem.draw(a)
        self.update(a,reward)
        self.history.append(reward)


def run_experiment(k,steps,agent_type,args):
    # An experiment consist of drawing a k-armed bandit problem and fitting a bandit algorithm over 1000 steps
    # the result of the experiment is the reward averaged over these 1000 steps
    #the experiments have high variance
    means=random.normal(0,1,k)
    stds=[1 for j in range(k)]
    pb=kArmedBanditProblem(k,means,stds)

    if agent_type=='AgentGreedy':
        agent=AgentGreedy(**args)
    if agent_type=='AgentUCB':
        agent=AgentUCB(**args)
    if agent_type=='AgentGrad':
        agent=AgentGrad(**args)

    for _ in range(steps):
        agent.play(pb)
    
    return(mean(agent.history))

def run_multiple_experiments(k,repeat,steps,agent_type,args):
    results=[]
    for _ in range(repeat):
        results.append(run_experiment(k,steps,agent_type,args))
    return mean(results)


def benchmarking_bandit(repeat=2000):
    k=10

    print("start")
    #Agent epsilon-greedy
    range_epsilon=[1/128,1/64,1/32,1/16,1/8,1/4]
    results_1=[]
    for epsilon in tqdm(range_epsilon):
        results_1.append(run_multiple_experiments(k,repeat=repeat,steps=2000,agent_type='AgentGreedy',
                                                args={'k':k,'q1':0,'epsilon':epsilon}))
    print("done (step 1)")

    #Agent greedy with apha=0.1 and optimistic init
    range_q=[1/8,1/4,1/2,1,2,4]
    results_2=[]
    for q in tqdm(range_q):
        results_2.append(run_multiple_experiments(k,repeat=repeat,steps=2000,agent_type='AgentGreedy',
                                                args={'k':k,'q1':q,'epsilon':0.0,'step_rule':'constant','alpha':0.1}))
    print("done (step 2)")

    #Agent UCB
    range_c=[1/32,1/16,1/8,1/4,1/2,1,2,4]
    results_3=[]
    for c in tqdm(range_c):
        results_3.append(run_multiple_experiments(k,repeat=repeat,steps=2000,agent_type='AgentUCB',
                                                args={'k':k,'q1':0,'c':c}))
    print("done (step 3)")

    #AgentGrad
    range_alpha=[1/32,1/16,1/8,1/4,1/2,1,2]
    results_4=[]
    for alpha in tqdm(range_alpha):
        results_4.append(run_multiple_experiments(k,repeat=repeat,steps=2000,agent_type='AgentGrad',args={'k':k,'alpha':alpha}))
    print("done (step 4)")

    import matplotlib.pyplot as plt
    # Plotting
    plt.figure(figsize=(12,10))
    plt.plot(range_epsilon, results_1, color='red', label=r'Agent $\epsilon$-greedy (parameter $\epsilon$)')
    plt.plot(range_q, results_2, color='black', label=r'Agent greedy with alpha=0.1 and optimistic init (parameter $Q_0$)')
    plt.plot(range_c, results_3, color='blue', label=r'Agent UCB (parameter $c$)')
    plt.plot(range_alpha, results_4, color='green', label=r'AgentGrad (parameter $\alpha$)')
    plt.xscale('log')
    plt.xlabel(xlabel=r'Parameter Value ($\epsilon$, $\alpha$, $c$, or $Q_0$)')
    plt.ylabel('Average Reward')
    plt.title('Comparison of Multi-Armed Bandit Algorithms')
    plt.xticks([1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4], ['$1/128$', '$1/64$', '$1/32$', '$1/16$', '$1/8$', '$1/4$', '$1/2$', '$1$', '$2$', '$4$'])
    plt.legend()
    plt.savefig('Reinforcement_Learning\Bandit_Algorithms\Benchmarking_Bandit.png')
    plt.show()

#benchmarking took ~1h for the whole 2000 experiments
if __name__=='__main__':
    benchmarking_bandit(repeat=2000)