import gym
import matplotlib.pyplot as plt

# env = gym.make('Acrobot-v1')
# env = gym.make('MountainCarContinuous-v0')
env = gym.make('MountainCar-v0')
# env = gym.make('HotterColder-v0')
# env = gym.make('CartPole-v1')
# for i_episode in range(20):
observation = env.reset()
for t in range(100):
    env.render()
    # print("observ:", observation)
    # action = env.action_space.sample()
    action = env.action_space.sample()
    print("action:", action)
    observation, reward, done, info = env.step(action)
    plt.pause(0.01)
    print("reward:", reward)
    # if done:
    #     print("Episode finished after {} timesteps".format(t + 1))
    #     break
env.close()
