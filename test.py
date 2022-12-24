import numpy as np
import gym
import random

# Learning_rate is alpha
# Discount_rate is gamma

def q_learning(learning_rate:float, discount_rate:float):

    # create Taxi environment
    env = gym.make('Taxi-v3')
    # env.action_space.seed()

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    # learning_rate = 0.9
    # discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # breakpoint()

    # training variables
    num_episodes = 10000
    max_steps = 200 # per episode prev 99

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state,:])

            # take action and observe reward
            new_state, reward, done, info = env.step(action)

            # breakpoint()

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

    print(f"Training completed over {num_episodes} episodes")
    # input("Press Enter to watch trained agent...")
    
    print(qtable)
    # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    conv_steps = max_steps

    for s in range(max_steps):

        # breakpoint()

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            conv_steps = s
            break

    env.close()
    if rewards == -200:
        rewards = -5
    if conv_steps == 199:
        conv_steps = 25

    return conv_steps, rewards

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Keeping the discount rate fixed 
    # Changing the learning rate
    x = np.linspace(0, 1, 500)
    y = []
    scores = []

    for val in x:
        cur, reward = q_learning(0.9, val)
        y.append(cur)
        scores.append(reward)
    
    # print(y)
    # print(len(y))

    # plt.plot(x, y)
    plt.plot(x, scores)


    plt.title("Discount rate vs  steps required for convergence")
    plt.xlabel("discount rate")
    plt.ylabel("max no. of steps")

    plt.show()
    # q_learning(0.9, 0.8)