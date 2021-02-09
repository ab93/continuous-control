from argparse import ArgumentParser
from collections import deque
from typing import List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from src.agent import Agent


RANDOM_SEED = 42
ENV_PATH = "../Reacher_Linux_NoVis/Reacher.x86_64"
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

    init_agent(state_size, action_size, num_agents)


def init_env(path):
    global env

    print("\nInitializing env....")
    env = UnityEnvironment(file_name=path)
    examine_env()


def init_agent(state_size, action_size, num_agents):
    global agent

    print("\nInitializing agent....")
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents,
                  random_seed=RANDOM_SEED)


def train_agent(n_episodes=1000, max_t=2500, print_every=100, noise_decay_factor=0.99, noise_wt_min=0.1):
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
        agent_scores = np.zeros(len(env_info.agents))

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
            print('\rEpisode {}\tEpisode Score: {:.2f}\t'
                  'Noise Factor: {:.2f}\tMax Score: {:.2f}\t'
                  'Average Score: {:.2f}'.format(i_episode, avg_agent_score, noise_wt,
                                             max_score, rolling_mean_score), end="")

        if rolling_mean_score >= BASELINE_SCORE:
            print('\nEnvironment solved in {:d} Episodes \t'
                  'Average Score: {:.2f}'.format(i_episode, rolling_mean_score))
            torch.save(agent.actor_local.state_dict(), 'weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'weights/checkpoint_critic.pth')
            break

    return scores


def plot_rewards(scores_: List[float]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_) + 1), scores_)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser(description="Train using DDPG algorithm")
    parser.add_argument("--path", dest="path", type=str, required=True,
                        help="Path of the Reacher Unity environment")
    parser.add_argument("--print_every", dest="print_every", type=int, default=50,
                        help="Print every given lines")

    args = parser.parse_args()

    init_env(args.path)

    scores = train_agent(print_every=args.print_every)
    plot_rewards(scores)

    env.close()
