# CartPole-v0
import gym
import math
from brain import DQNAgent

#def modify_reward(observation, x_th, theta_th):
def modify_reward(observation):
    x_th = 2.4
    theta_th = 12 * 2 * math.pi / 360
    x, x_dot, theta, theta_dot = observation
    r1 = (x_th - abs(x)) / x_th
    r2 = (theta_th - abs(theta)) / theta_th
    return r1 * r2

if __name__ == "__main__":
    # New environment
    env = gym.make('CartPole-v0')
    #env = env.unwrapped # To access inner variables like x_threshold and theta_threshold_radians

    # Record the video
    directory = 'video/'
    env = gym.wrappers.Monitor(env, directory, force=True)

    # New agent
    agent = DQNAgent(
        n_actions = env.action_space.n,
        n_states = env.observation_space.shape[0],
        batch_size = 32,
        learning_rate = 0.01,
        epsilon = 0.9,
        gamma = 0.9,
        target_replace_iter = 100,
        replay_memory_size = 2000,
        output_graph = True
    )

    for i_episode in range(400):
        # Initial observation
        observation = env.reset()

        t = 0
        score = 0
        while True:
            env.render()

            # Choose action based on observation
            action = agent.choose_action(observation)

            # Take action
            observation_, reward, done, info = env.step(action)

            # Modify the reward
            #reward = modify_reward(observation_, env.x_threshold, env.theta_threshold_radians)
            reward = modify_reward(observation_)

            ''' The score for CartPole-v0 is the cumulative sum of the rewards
                but in this case the reward is modified. Just look `t` if you
                want to know the ufficial score. '''
            # Update score
            score += reward

            # Store transition
            agent.store_transition(observation, action, reward, observation_)

            # Learn from experience
            agent.learn()

            # Swap observations
            observation = observation_
            t += 1

            if done:
                print 'Ep: {0:3d} steps: {1:3d} score: {2:4.2f}'.format(i_episode, t, score)
                break
