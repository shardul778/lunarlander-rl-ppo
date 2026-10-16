import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json, time, math, traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


class HoverBonus(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, vang, l, r = obs
        if abs(angle) < 0.2 and abs(vx) < 0.2 and abs(vy) < 0.2:
            reward += 1.0
        return obs, reward, terminated, truncated, info


class ProgressShaping(gym.Wrapper):
    def __init__(self, env, penalty=0.05, scale=10.0):
        super().__init__(env)
        self.penalty, self.scale, self.prev_phi = penalty, scale, 0.0

    @staticmethod
    def phi(obs):
        x, y, vx, vy, angle = obs[:5]
        return -(abs(x) + abs(vx) + abs(vy) + abs(angle))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.prev_phi = self.phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        p = self.phi(obs)
        reward += self.scale * (p - self.prev_phi) - self.penalty
        self.prev_phi = p
        return obs, reward, terminated, truncated, info


CONFIGS = {"hover": HoverBonus, "fixed": ProgressShaping}
SEEDS = [0, 1, 2]
BUDGET_MIN = 30
N_ENVS = 8
ROLLOUT = 1024 * N_ENVS
PPO_KW = dict(n_steps=1024, batch_size=256, n_epochs=4, gamma=0.999,
              gae_lambda=0.98, ent_coef=0.01, device="cpu", verbose=0)


def measure_sps(n=2 * ROLLOUT):
    torch.set_num_threads(1)
    env = make_vec_env("LunarLander-v3", n_envs=N_ENVS)
    model = PPO("MlpPolicy", env, seed=0, **PPO_KW)
    t = time.time()
    model.learn(n)
    sps = n / (time.time() - t)
    env.close()
    return sps


def train_one(name, seed, steps):
    try:
        torch.set_num_threads(1)
        env = make_vec_env("LunarLander-v3", n_envs=N_ENVS,
                           wrapper_class=CONFIGS[name], seed=seed)
        model = PPO("MlpPolicy", env, seed=seed, **PPO_KW)
        model.learn(steps)
        model.save(f"models/ppo_{name}_s{seed}")
        env.close()

        ev = Monitor(gym.make("LunarLander-v3"))
        rets, lens = evaluate_policy(model, ev, n_eval_episodes=20,
                                     deterministic=True, return_episode_rewards=True)
        rets, lens = np.array(rets), np.array(lens)
        return dict(name=name, seed=seed, steps=steps,
                    mean_return=float(rets.mean()),
                    success_rate=float((rets >= 200).mean()),
                    hover_rate=float((lens >= 1000).mean()))
    except Exception:
        return dict(name=name, seed=seed, error=traceback.format_exc())


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    jobs = [(n, s) for n in CONFIGS for s in SEEDS]
    cores = os.cpu_count() or 1
    workers = min(len(jobs), cores)
    waves = math.ceil(len(jobs) / workers)

    sps = measure_sps()
    per_run_sec = BUDGET_MIN * 60 * 0.8 / waves
    steps = int(sps * per_run_sec * 0.7)
    steps = max(ROLLOUT, steps // ROLLOUT * ROLLOUT)
    print(f"cores={cores} workers={workers} waves={waves} "
          f"speed={sps:.0f} steps/s -> {steps:,} steps per run", flush=True)

    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(train_one, n, s, steps) for n, s in jobs]
        for f in as_completed(futs):
            r = f.result()
            if "error" in r:
                print(f"FAILED {r['name']} s{r['seed']}:\n{r['error']}", flush=True)
                continue
            results.append(r)
            print(f"done {r['name']} s{r['seed']}: ret={r['mean_return']:.1f} "
                  f"success={r['success_rate']:.0%} hover={r['hover_rate']:.0%} "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    for name in CONFIGS:
        rs = [r for r in results if r["name"] == name]
        if not rs:
            continue
        print(f"{name:6s} return {np.mean([r['mean_return'] for r in rs]):7.1f} "
              f"± {np.std([r['mean_return'] for r in rs]):5.1f} | "
              f"success {np.mean([r['success_rate'] for r in rs]):.0%} | "
              f"hover {np.mean([r['hover_rate'] for r in rs]):.0%}")
