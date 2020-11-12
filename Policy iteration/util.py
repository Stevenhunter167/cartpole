import numpy as np
np.random.seed(0)

def inferRandom():
    return np.random.randint(0, 2)

def playGame(env, agent, exploration=0, maxstep=500, render=False):
    # output data
    observation_action = []

    # start the game
    observation = env.reset()
    for t in range(maxstep):

        if render:
            env.render()

        # introduce randomness
        if np.random.rand() > exploration:
            action = agent.infer(observation)
        else:
            action = inferRandom()
        
        # record the observation-action pair
        observation_action.append(list(observation) + [action])

        # get the next tick of game
        observation, reward, done, info = env.step(action)

        # record the reward

        if done:
            break
    
    total_reward = t + 1
    return observation_action, total_reward

def evaluate(env, learner, size, render=False):
    totalrewards = []
    actions = []
    for i_episode in range(size):
        obs_act, reward = playGame(env, learner, exploration=0, render=render)
        # print(f'episode {i_episode} reward {reward}')
        actions.append(obs_act[-1])
        totalrewards.append(reward)
    return totalrewards, np.mean(actions)

def generateDataBatch(env, agent, size, exploration=0, criteria=50):
    # variables for accepted
    obs_acts = []
    acc_rwd = []

    # all games
    currentgames = 0
    rewardstats = []

    while currentgames < size:
        obs_act, cur_reward = playGame(env, agent, exploration=exploration)
        print(f"cur_reward {cur_reward}, {len(acc_rwd)} accepted")
        rewardstats.append(cur_reward)

        if cur_reward > criteria:
            obs_acts.extend(obs_act)
            acc_rwd.append(cur_reward)
            currentgames += 1
    
    return np.array(obs_acts), acc_rwd, rewardstats