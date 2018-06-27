from gym.envs.registration import registry, register, make, spec

register(
    id='RocketLander-v4',
    entry_point='gym.envs.box2d:RocketLander',
    max_episode_steps=1000,
    reward_threshold=0,
)
