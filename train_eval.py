import os
import torch
from stable_baselines3 import DQN
from envs import make_context_env


def eval_DQN(sumo_env, model,  max_iter = 1000):
    obs, info = sumo_env.reset()
    total_reward = 0
    info_dump = []
    for _ in range(max_iter):
        with torch.no_grad():
            action, meta = model.predict(obs)
        obs, reward, done, trunc, info = sumo_env.step(action)
        total_reward += reward
        info['reward'] = reward
        info_dump.append(info)
        if done:
            break
    return total_reward, info_dump

def train_DQN(ctx, reset=False):
    context, sumo_env = make_context_env(ctx, reset=reset)
    sumo_env.reset()
    if reset or (not os.path.exists(f'./experts/dqn{context.name}.pth')):
        model = DQN(
            env=sumo_env,
            policy="MlpPolicy",
            learning_rate=1e-3,
            learning_starts=0,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            verbose=1,
            device = torch.device('mps')
        )
        model.learn(total_timesteps=10000)
        model.save(f'./experts/dqn{context.name}.pth')
    else:
        model = DQN.load(f'./experts/dqn{context.name}.pth')
    return model, sumo_env