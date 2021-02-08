from collections import deque
from typing import List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from agent import Agent


RANDOM_SEED = 42
ENV_PATH = "Reacher_Linux_NoVis_v2/Reacher.x86_64"
BASELINE_SCORE = 30.0

env: Optional[UnityEnvironment] = None
agent: Optional[Agent] = None


def get_brain(env_: UnityEnvironment) -> tuple:
    brain_name = env_.brain_names[0]
    brain = env_.brains[brain_name]

    return brain_name, brain


def examine_env():
    global env

    print("\nExamining env....")
    brain_name, brain = get_brain(env)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def init_env():
    global env

    print("\nInitializing env....")
    env = UnityEnvironment(file_name=ENV_PATH)
    examine_env()


def init_agent():
    global agent

    print("\nInitializing agent....")
    agent = Agent(state_size=33, action_size=4, random_seed=42)


def train_agent(n_episodes=1000, max_t=1000, print_every=100, noise_decay_factor=0.99, noise_wt_min=0.1):
    global env, agent

    brain_name, _ = get_brain(env)

    print("\nStarting to train the agent....\n")

    rolling_scores = deque(maxlen=print_every)
    scores = []
    noise_factor = 1.0

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agent.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]

            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            is_done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, is_done)
            state = next_state
            score += reward
            if is_done:
                break

        rolling_scores.append(score)
        scores.append(score)

        rolling_mean_score = np.mean(rolling_scores)

        print('\rEpisode {}\tEpisode Score: {:.2f}\t'
              'Average Score: {:.2f}'.format(i_episode, score, rolling_mean_score), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, rolling_mean_score))

        if rolling_mean_score >= BASELINE_SCORE:
            print('\nEnvironment solved in {:d} Episodes \t'
                  'Average Score: {:.2f}'.format(i_episode, rolling_mean_score))
            torch.save(agent.actor_local.state_dict(), 'weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'weights/checkpoint_critic.pth')
            break

    return scores


def train_multi_agents(n_episodes=1000, max_t=10000, print_every=100, noise_decay_factor=0.99, noise_wt_min=0.1):
    global env, agent

    brain_name, _ = get_brain(env)

    print("\nStarting to train the agent....\n")

    rolling_scores = deque(maxlen=print_every)
    scores = []
    noise_wt = 1.0
    max_score = 0.0

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        agent_scores = np.zeros(20)

        for t in range(max_t):
            actions = agent.act(states, weight=noise_wt)
            env_info = env.step(actions)[brain_name]

            rewards = env_info.rewards
            next_states = env_info.vector_observations
            is_dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, is_dones)
            states = next_states
            agent_scores += rewards
            if np.any(is_dones):
                break

        avg_agent_score = np.mean(agent_scores)
        max_score = max(max_score, avg_agent_score)
        rolling_scores.append(avg_agent_score)
        scores.append(avg_agent_score)
        noise_wt = max(noise_decay_factor * noise_wt, noise_wt_min)

        rolling_mean_score = np.mean(rolling_scores)

        print('\rEpisode {}\tEpisode Score: {:.2f}\t'
              'Noise Factor: {:.2f}\tMax Score: {:.2f}\t'
              'Average Score: {:.2f}'.format(i_episode, avg_agent_score, noise_wt,
                                             max_score, rolling_mean_score), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, rolling_mean_score))

        if rolling_mean_score >= BASELINE_SCORE:
            print('\nEnvironment solved in {:d} Episodes \t'
                  'Average Score: {:.2f}'.format(i_episode, rolling_mean_score))
            torch.save(agent.actor_local.state_dict(), 'weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'weights/checkpoint_critic.pth')
            break

    return scores


def plot_rewards(scores: List[float]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


if __name__ == "__main__":
    init_env()
    init_agent()

    scores = train_multi_agents(print_every=50)
    plot_rewards(scores)

    env.close()
