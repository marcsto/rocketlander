"""
  Allows you play the rocker lander environment using the keyboard.
  
  Up arrow: Increase throttle
  Down arrow: Decrease throttle
  Left / Right arrow: Move the throttle left and right
  1 and 2: Use left and right control thrusters
  
  Credit: This is based on:
  https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py
"""

from gym.envs.registration import register
import gym
import time

register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
)

env = gym.make("RocketLander-v0")

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 6
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    
    print(key)
    if key==65362:#up
        human_agent_action = 2
    elif key==65364:#down
        human_agent_action = 3
    elif key==65361:#left
        human_agent_action = 1
    elif key==65363:#right
        human_agent_action = 0
    elif key==ord('1'):#1
        human_agent_action = 4
    elif key==ord('2'):#space
        human_agent_action = 5
        

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    print("Release...", a)
    #if a <= 0 or a >= ACTIONS: return
    #if human_agent_action == a:
    human_agent_action = 6

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    _obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        _obser, r, done, _info = env.step(a)
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break