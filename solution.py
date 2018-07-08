"""
  One solution to the rocket lander environment. Uses Proximal Policy Optimization.

  Author: Marc Stogaitis
""" 

# Run these commands to download the dependencies before running:
# pip -q install scipy tqdm joblib zmq dill progressbar2 cloudpickle opencv-python
# pip -q install --no-deps git+https://github.com/openai/baselines.git@24fe3d6576dd8f4cdd5f017805be689d6fa6be8c

from gym.envs.registration import register, make
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import gym
import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies

# Register the RockerLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
)

def make_env(model_num, proc_idx=0):
    def make():
        env = gym.make('RocketLander-v0')
        env = gym.wrappers.Monitor(env, "../run" + str(model_num) + "/" + str(proc_idx) + "/", force=True)
        return env
    return make

# Default hyper parameters. Can modify by adding key : values in the experiment.
class HyperParams(object):
    def __init__(self):
        self.model = -1
        self.nsteps=1024
        self.nminibatches=256
        self.lam=0.95
        self.gamma=0.99
        self.noptepochs=3
        self.log_interval=1
        self.ent_coef=0.01
        self.lr=1e-4
        self.cliprange=0.2
        self.env_count=1

# Easy way to define different experiments. Data will be stored in the model dir.
experiment = [
    {'model': 1, 'env_count':12}, 
]

hparams = HyperParams()
original_len = len(hparams.__dict__)
hparams.__dict__.update(experiment[0])
if original_len != len(hparams.__dict__):
    print("Invalid hyper parameter")
    raise 

if hparams.env_count == 1: 
    vec_env = DummyVecEnv([make_env(hparams.model)])
else:
    envs = []
    for i in range(hparams.env_count):
        envs.append(make_env(hparams.model, i))
    vec_env = SubprocVecEnv(envs)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config):
    ppo2.learn(policy=policies.MlpPolicy,
             env=vec_env,
             nsteps=hparams.nsteps,
             nminibatches=hparams.nminibatches,
             lam=hparams.lam,
             gamma=hparams.gamma,
             noptepochs=hparams.noptepochs,
             log_interval=hparams.log_interval,
             ent_coef=hparams.ent_coef,
             lr=lambda _: hparams.lr,
             cliprange=lambda _: hparams.cliprange,
             total_timesteps=int(2e8))
