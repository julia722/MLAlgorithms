# q_learning.py (jmkim2) HW 8

from environment import MountainCar
import sys
import numpy as np

class Model():
    def __init__(self, car, episodes, max_iters, epsilon, gamma, lr):
        self.car = car
        self.episodes = int(episodes)
        self.max_iters = int(max_iters)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.lr = float(lr)
        
        self.actions = car.action_space
        self.states = car.state_space
        self.weight = np.zeros(shape=(self.states, self.actions))
        self.bias = 0
        
    def train(self):
        returns = []
        for episode in range(self.episodes):
            state = self.car.reset() # reset environment to starting conditions
            r = 0; i = 0
            done = False
            
            while not done and i < self.max_iters:
                qval = self.qval(state, self.weight, self.bias)
                action = self.select_action(state, self.epsilon, qval, self.actions)
                new_state, reward, done = self.car.step(action)
                r += reward

                qval_new = self.qval(new_state, self.weight, self.bias)
                qval_max = np.max(qval_new)

                optimal = reward + self.gamma * qval_max
                gradient = self.grad(state, action, self.car, self.actions, self.states)
                # update weight and bias
                self.weight -= self.lr * (qval[action] - optimal) * gradient
                self.bias -= self.lr * (qval[action] - optimal)
                state = new_state
                i += 1
            returns.append(r)
        return self.weight, self.bias, returns

    def qval(self, state, weight, bias):
        val = bias # add in bias
        for key in state:
            val += weight[key] * state[key]
        return val
    
    def grad(self, state, action, car, actions, states):
        gradient = np.zeros(shape=(states, actions))
        update = np.zeros(states)
        for key in state:
            update[key] = state[key]
        gradient[:, action] = update
        return gradient

    def select_action(self, state, e, qval, actions):
        if np.random.uniform(0, 1) < 1 - e: # if there is a draw, choose the smallest action
            return np.argmax(qval)
        else: # choose randomly from 0, 1, 2
            return np.random.randint(actions)

def main():
    mode = sys.argv[1] # raw or tile
    car = MountainCar(mode)
    episodes = sys.argv[4]
    max_iters = sys.argv[5]
    epsilon = sys.argv[6]
    gamma = sys.argv[7]
    lr = sys.argv[8]

    # train model
    m = Model(car, episodes, max_iters, epsilon, gamma, lr)
    weights, bias, returns = m.train()
    weights = np.reshape(weights, (car.state_space*car.action_space))

    # metrics out
    weight_out = open(str(sys.argv[2]), "w")
    weight_out.write(str(bias) + "\n")
    for w in weights:
        weight_out.write(str(w) + "\n")
    weight_out.close()

    returns_out = open(str(sys.argv[3]), "w")
    for r in returns:
        returns_out.write(str(r) + "\n")
    returns_out.close()
    print("done")


if __name__ == "__main__":
    main()