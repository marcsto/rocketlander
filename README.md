### Easy way to run OpenAi Gym Rocketlander (aka Falcon Rocket) in your browser
Ever wanted to write code that lands a SpaceX-like rocket on a platform in the ocean? Now you can try it directly from your browser by using the following colab notebook:

https://colab.research.google.com/drive/1gFdJueFbLYBPuI8ss_0WHG0VOXj2SNdT#scrollTo=phVnskT-RKXr

At the time of writing, Google provides a free virtual machine with a K80 GPU to train on.

### Credit
This is a copy of https://github.com/EmbersArc/gym that only has the rocket_lander.py file to make it easy to git pull into an existing gym environment.

### Local Usage:
If you want to run the environment on your own machine instead of in the colab notebook, follow these steps:

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
          
Please see EmbersArc's excellent repository for more details on the rocket lander gym environment
