import numpy.random as rand
import numpy as np

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