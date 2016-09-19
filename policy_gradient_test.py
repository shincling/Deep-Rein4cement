import numpy.random as rand
import numpy as np

def theta_policy(n_states,n_actions):
    '''theta --- policy for state i and action j'''
    theta=np.zeros((n_states,n_actions))
    return theta

theta=theta_policy(5,2)
print theta
