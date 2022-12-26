import numpy as np
import gym
import random
import matplotlib.pyplot as plt

def main():

    # create Taxi environment
    env = gym.make('Taxi-v3')
    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99 # per episode

    # initialize empty lists to store Q values and rewards
    q_values = []
    rewards = []
    penalties = [] # initialize empty list to store steps per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False
        episode_reward = 0

        p = 0

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

            if reward == -10:
                p += 1

            episode_reward += reward
            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Update to our new state
            state = new_state

            # append current Q value to list
            q_values.append(np.max(qtable[state,:]))

            # if done, finish episode
            if done == True:
                break
        penalties.append(p)

        # append total reward and steps for episode to rewards and steps lists
        rewards.append(episode_reward)
        # steps.append(s+1) # add 1 to include the starting step

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    done = False

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        env.render()
        
        state = new_state

        if done == True:
            break

    env.close()

    # calculate average reward per episode
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward per episode: {avg_reward}")

    # calculate success rate in achieving the goal
    success_rate = sum([1 for r in rewards if r > 0]) / len(rewards)
    print(f"Success rate in achieving the goal: {success_rate}")

    # plot number of steps vs number of episodes
    plt.title("Penalties vs number of episodes")
    plt.plot(penalties)
    plt.xlabel("Episode")
    plt.ylabel("Penalties")
    plt.show()
    
    
    # plot Q values values as y and iteration as x
    # plt.plot(q_values)
    # plt.xlabel("Iteration")
    # plt.ylabel("Q Value")
    # plt.show()
    # plot rewards
    # plt.plot(rewards)
    # plt.xlabel("Iteration")
    # plt.ylabel("Reward")
    # plt.show()
if __name__ == "__main__":
    main()
