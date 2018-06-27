# OpenAi Gym Rocketlander
Copy of https://github.com/EmbersArc/gym that only has the rocket_lander.py file to make it easy to git pull into an existing gym environment.

Please see EmbersArc's excellent repository for more details

#### Usage:
**Install openai gym with box2d support**
    
    pip install 'gym[box2d]'

**Clone this repo**

    git clone https://github.com/marcsto/rocketlander.git

**Add the following to your python code to register the environment**

    from gym.envs.registration import registry, register, make, spec
    register(
        id='RocketLander-v0',
        entry_point='rocketlander.rocket_lander:RocketLander',
        max_episode_steps=1000,
        reward_threshold=0,
    )

This assumes that this repo (the rocketlander dir) is a sub directory of where your python code is.
    
    i.e.:
          run_python.py
          rocketlander/rocket_lander.py
