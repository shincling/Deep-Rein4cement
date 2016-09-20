# coding=utf8
import numpy.random as rand
import numpy as np
import mdptoolbox
import mdptoolbox.example as mdp_ex

def generate_rollout(mu_policy, transition_matrix, reward_matrix, discount, path_len=10) :
    n_states, n_actions = mu_policy.shape
    path_action = np.zeros(path_len , dtype=int)
    path_state = np.zeros(path_len, dtype=int)
    path_state[0] = rand.random_integers(n_states) -1
    path_action[0] = np.random.choice(n_actions, 1, p=mu_policy[path_state[0]])
    path_reward =  reward_matrix[path_state[0], path_action[0]]
    for i in range(1, path_len) :
        path_state[i] = np.random.choice(n_states , 1, p=transition_matrix[path_action[i-1]][path_state[i-1]])#这里是５个state里选一个，选择的概率用一个１×5的array衡量，在这里是[1,0,0,0,0]
        path_action[i] = np.random.choice(n_actions, 1, p=mu_policy[path_state[i]])
        path_reward =  path_reward + discount**i * reward_matrix[path_state[i], path_action[i]] #reward是之前的和这次的乘以折扣之后的累加
    return path_state, path_action, path_reward


def generate_rollouts(mu_policy, transition_matrix, reward_matrix, discount, path_len=10, n_paths=10):
    n_states, n_actions = mu_policy.shape
    paths_states = np.zeros(( n_paths , path_len), dtype=int)
    paths_actions = np.zeros(( n_paths , path_len), dtype=int)
    paths_rewards = np.zeros( n_paths  )
    for i in range(0, n_paths) :
        paths_states[i], paths_actions[i], paths_rewards[i] = generate_rollout(mu_policy, transition_matrix, reward_matrix, discount, path_len)
    return paths_states, paths_actions, paths_rewards





def theta_policy(n_states,n_actions):
    '''theta --- policy for state i and action j'''
    theta=np.zeros((n_states,n_actions))
    return theta

theta=theta_policy(5,2)
print theta

def mu_policy(theta):
    '''mu ---　softmax policy'''
    n_states,n_actions=theta.shape
    mu=np.zeros((n_states,n_actions))
    for state in range(0,n_states):
        max_theta=np.max(theta[state])
        mu[state]=np.exp(theta[state]-max_theta)/np.sum(np.exp(theta[state]-max_theta))
    return mu

mu=mu_policy(theta)
print mu

def log_mu_policy(theta):
    '''mu ---　softmax policy'''
    n_states,n_actions=theta.shape
    log_mu=np.zeros((n_states,n_actions))
    for state in range(0,n_states):
        max_theta=np.max(theta[state])
        log_sum_exp=max_theta+np.log(np.sum(np.exp(theta[state])))-max_theta
        log_mu[state]=theta[state]-log_sum_exp
    return log_mu
log_mu=log_mu_policy(theta)
print log_mu

def gradient_log_mu_policy(theta,mu,state,action):
    n_states,n_actions=theta.shape
    grad=np.zeros((n_states,n_actions))
    grad[state,action]=1
    max_theta=np.max(theta[state])
    grad[state]=grad[state]-mu[state]
    return grad
print "gradient log mu with state-1 action-1:\n",gradient_log_mu_policy(theta,mu,1,1)
print "mu policy after adding gradient:\n",mu_policy(theta+gradient_log_mu_policy(theta,mu,1,1))

def path_likelihood_ratio(theta_policy,mu_policy,path_states,path_actions):
    likehood_ratio=np.zeros(theta_policy.shape)
    path_len=len(path_states)
    for t in range(0,path_len):
        likehood_ratio=likehood_ratio+gradient_log_mu_policy(theta_policy,mu_policy,path_states[t],path_actions[t])
    return likehood_ratio

def policy_gradient(theta_policy, mu_policy , paths_states , paths_actions, paths_rewards) :
    policy_grad = np.zeros(theta_policy.shape)
    n_paths , path_len =  paths_states.shape
    for path in range(0, n_paths) :
        policy_grad = policy_grad + paths_rewards[path]*path_likelihood_ratio(theta_policy, mu_policy, paths_states[path], paths_actions[path])
    policy_grad = policy_grad/n_paths
    return policy_grad

def policy_gradient_algo(transitions, rewards, discount, path_len=10,  n_paths=100, gamma=1.0, eps=0.01, n_iterations=100, logging=False) :
    n_states, n_actions = rewards.shape
    theta = theta_policy(n_states, n_actions)
    mu = mu_policy(theta)
    n=0
    paths_states, paths_actions, paths_rewards = generate_rollouts(mu, transitions, rewards, discount , path_len, n_paths )
    pgrad = policy_gradient(theta, mu,  paths_states, paths_actions, paths_rewards)
    theta_diff =  (gamma/(n+1))*pgrad
    theta_diff_norm = np.linalg.norm(theta_diff)
    #mu_diff = np.linalg.norm(mu_policy(theta) - mu_policy(theta+pgrad))
    while ( (n<n_iterations) & (theta_diff_norm>eps) ):
        if (logging) :
            print "mu policy : \n",  mu_policy(theta)
            print "policy gradient: \n", pgrad
            print "theta policy : \n" , theta_diff
            print "theta policy diff norm: " , theta_diff_norm
        theta_diff =  (gamma/(n+1))*pgrad
        theta = theta + theta_diff
        mu = mu_policy(theta)
        paths_states, paths_actions, paths_rewards = generate_rollouts(mu_policy(theta), transitions, rewards, discount , path_len, n_paths )
        pgrad = policy_gradient(theta, mu , paths_states, paths_actions, paths_rewards)
        #mu_diff = np.linalg.norm(mu_policy(theta) - mu_policy(theta+pgrad))
        theta_diff_norm = np.linalg.norm(theta_diff)
        n = n+1
    return theta

n_states = 5
n_actions = 2
fire_prob = 0.1
discount=0.9
n_paths=100
path_len=100
path_len = 10
path_num = 10

P, R = mdp_ex.forest(S=n_states, p=fire_prob)
#P是转移矩阵，大小是action*state*state,(a,i,j)的意思是在状态i下采用a转移到j状态的概率
#R是reward矩阵，大小是action*state,(a,i)的意思是在状态i下瓷用a得到的reward是多少


pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=discount)
pi.policy0=[1,1,1,1,1]
#vi.setVerbose()
pi.run()

policy_pi = pi.policy

print "Optimal policy (policy iteration) : \n" , policy_pi

policy_pg  = policy_gradient_algo( P, R , discount , path_len ,  n_paths, gamma=10 , eps=0.01)

print "Optimal policy (policy gradient) :\n" , mu_policy(policy_pg)


