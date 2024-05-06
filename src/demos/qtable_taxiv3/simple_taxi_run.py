# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 R. Berkay Bozkurt <resitberkaybozkurt@gmail.com>

# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import datetime
import json
import random
from typing import Optional, Tuple

import gymnasium as gym
import imageio
import numpy as np
import pickle5 as pickle
import tqdm
from simple_taxi_train import greedy_policy
from taxi_custom import customTaxiEnv
from tqdm.notebook import tqdm

# Create environment and read Q-table
# env = gym.make("Taxi-v3", render_mode="rgb_array")
env = customTaxiEnv(render_mode="rgb_array")
state_space = env.observation_space.n
action_space = env.action_space.n
q_table_file = "src/demos/qtable_taxiv3/q_table_taxi.pkl"

# Load the Q-table from disk
with open(q_table_file, "rb") as f:
    Qtable_taxi = pickle.load(f)

print("Q-table loaded from:", q_table_file)


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param max_steps: Maximum number of steps per episode
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, Qtable, out_directory, max_frames=100, fps=1):
    """
    Generate a replay video of the agent.

    :param env: OpenAI Gym environment
    :param Qtable: Q-table of our agent
    :param out_directory: Path to save the recorded video
    :param max_frames: Maximum number of frames to record
    :param fps: Frames per second for the recorded video
    """
    images = []
    terminated = False
    truncated = False
    # state, info = env.reset(seed=random.randint(0, 500))
    passenger_location_index = 2  # Set the starting location of the passenger
    state, info = env.custom_reset(passenger_location_index)
    img = env.render()
    images.append(img)
    frame_count = 0  # Counter to keep track of recorded frames
    while not terminated and not truncated and frame_count < max_frames:
        # Take the action (index) that has the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)
        frame_count += 1
    imageio.mimsave(
        out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps
    )


def make_movie(model, env, video_fps=1, local_repo_path="/"):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub

    :param model: Dictionary containing model information
    :param env: OpenAI Gym environment
    :param video_fps: Frames per second to record the video replay
    :param local_repo_path: Local repository path
    """
    eval_env = env

    # Step 4: Evaluate the model and build JSON with evaluation metrics
    mean_reward, std_reward = evaluate_agent(
        eval_env,
        model["max_steps"],
        model["n_eval_episodes"],
        model["qtable"],
        model["eval_seed"],
    )

    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

    # Write a JSON file called "results.json" that will contain the evaluation results
    with open("results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 6: Record a video
    video_path = "src/demos/qtable_taxiv3/replay.mp4"
    record_video(env, model["qtable"], video_path, max_frames=100, fps=video_fps)

    print("Video recorded successfully at:", video_path)


# Environment parameters
env_id = "Taxi-v3"  # Name of the environment
max_steps = 200  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob
# Training parameters
n_training_episodes = 100000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 99  # Total number of test episodes
eval_seed = [
    16,
    54,
    165,
    177,
    191,
    191,
    120,
    80,
    149,
    178,
    48,
    38,
    6,
    125,
    174,
    73,
    50,
    172,
    100,
    148,
    146,
    6,
    25,
    40,
    68,
    148,
    49,
    167,
    9,
    97,
    164,
    176,
    61,
    7,
    54,
    55,
    161,
    131,
    184,
    51,
    170,
    12,
    120,
    113,
    95,
    126,
    51,
    98,
    36,
    135,
    54,
    82,
    45,
    95,
    89,
    59,
    95,
    124,
    9,
    113,
    58,
    85,
    51,
    134,
    121,
    169,
    105,
    21,
    30,
    11,
    50,
    65,
    12,
    43,
    82,
    145,
    152,
    97,
    106,
    55,
    31,
    85,
    38,
    112,
    102,
    168,
    123,
    97,
    21,
    83,
    158,
    26,
    80,
    63,
    5,
    81,
    32,
    11,
    28,
    148,
]  # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
# Each seed has a specific starting state


model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,
    "qtable": Qtable_taxi,
}


make_movie(model=model, env=env)
