##########################################
#             DISCLAIMER                 #
#                                        #
# This code is based largely on openai's #
# baselines code for deepq.              #
#                                        #
# https://github.com/openai/baselines    #
#                                        #
##########################################


#module imports
import gym
import numpy as np
import os
import dill
import pickle
import tempfile
import tensorflow as tf
import zipfile

#local imports
import logger
import models
import tf_util as U

#local function imports
from schedules import LinearSchedule
from build_graph import build_act, build_train
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame

#Defines the folder where the results will be saved
folder = 'Expirements_penalty/'

#Number of the expirenment
expirement = 'exp_3'

#name of envirenment
environment = 'Seaquest'
env_name = environment+'NoFrameskip-v4'

#Builds the file name for where the model and stats will be stored
name = environment+'_'+expirement
model_name = name+'_model'
rewards_name = name+'_rewards.pkl'
description_name = name+'.description'
model_file = os.path.join(folder, model_name)
rewards_file = os.path.join(folder, rewards_name)
description_file = os.path.join(folder, description_name)
model_saved = False


#Parameters of Q-learning that could potentialy be tweaked
lr=1e-4
max_timesteps=2000000
buffer_size=10000
exploration_fraction=0.1
exploration_final_eps=0.01
train_freq=4
learning_starts=10000
target_network_update_freq=1000
gamma=0.99
prioritized_replay=True

#Parameters of Q learning that are usualy kept as default
batch_size=32
print_freq=2
checkpoint_freq=10000
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=None
prioritized_replay_eps=1e-6
num_cpu=16
param_noise=False
callback=None

#A description text for future reference
objective = 'Q-learning with penalty for non null actions.'
action_penalty = 0.1

description ="""
###############################################################################
#                          Expirement Description                             #
#                                                                             #
# Envirionment: {1:20s}{0:>43} 
# Expirment Number: {2:10s}{0:>49}
# Destination Folder: {3:40s}{0:>17}
#                                                                             #
# Objective: {4:60s}{0:>6}             
#                                                                             #
# '''''''''''''''''''''''                                                     #
# 'Experiment parameters'                                                     #
# '''''''''''''''''''''''                                                     #
#                                                                             #
# - Action Penalty: {25:<6f}{0:>53}
# - Learning Rate: {5:<5g}{0:>54}
# - Number of Time Steps: {6:<20d}{0:>33}
# - Replay Memory Size: {7:<15d}{0:>40}
# - Exploration Fraction: {8:<5f}{0:>45}
# - Exploration Final Epsilon: {9:<5f}{0:>40}
# - Train Frequency: {10:<5d}{0:>53}
# - Learning Starts at: {11:<10d}{0:>45}
# - Target Network Update frequency: {12:<10d}{0:>32}
# - Gamma: {13:<.3f}{0:>63}
# - Prioritized Replay: {14:10s}{0:>45}
#{0:>78} 
# - Batch Size = {15:<5d}{0:>57}
# - Print Frequency = {16:<5d}{0:>52}
# - Checkpoint Frequency = {17:<5d}{0:>47}
# - Prioritized Replay alpha = {18:<5f}{0:>40}
# - Prioritized Replay beta = {19:<5f}{0:>41}
# - Prioritized replay beta iters = {20}{0:>39}
# - Prioritized replay eps = {21:<5g}{0:>45}
# - num cpu? = {22:<3d}{0:>61}
# - Parameter Noise = {23:10s}{0:>47}
# - callback = {24}{0:>60}
#                                                                             #
###############################################################################
"""

description = description.format('#',environment, expirement, folder, objective, 
                                 lr, 
                                 max_timesteps,
                                 buffer_size,
                                 exploration_fraction,
                                 exploration_final_eps,
                                 train_freq,
                                 learning_starts,
                                 target_network_update_freq,
                                 gamma,
                                 str(prioritized_replay),
                                 batch_size,
                                 print_freq,
                                 checkpoint_freq,
                                 prioritized_replay_alpha,
                                 prioritized_replay_beta0,
                                 prioritized_replay_beta_iters,
                                 prioritized_replay_eps,
                                 num_cpu,
                                 str(param_noise),
                                 callback,
                                 action_penalty
                                )

print(description)
with open(description_file, 'w') as f:
    f.write(description)
input()


#makes the environment, and wraps it with a couple of preprocessing layers.
env = gym.make(env_name)
env = ScaledFloatFrame(wrap_dqn(env))

#builds the model for the q_function
model = models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )
q_func=model

#function for creating a placeholder for the observation
#Not entirely sure why a separate function is needed for this
def make_obs_ph(name):
    return U.BatchInput(env.observation_space.shape, name=name)

#This function returns a few funcitons which are used later on in the training
act, train, update_target, debug = build_train(
    make_obs_ph=make_obs_ph,
    q_func=q_func,
    num_actions=env.action_space.n,
    optimizer=tf.train.AdamOptimizer(learning_rate=lr),
    gamma=gamma,
    grad_norm_clipping=10,
    param_noise=param_noise
)

#act parameters
#No idea what this is used for
act_params = {
    'make_obs_ph': make_obs_ph,
    'q_func': q_func,
    'num_actions': env.action_space.n,
}


#Creates the replay buffer, Prioritized or otherwise
if prioritized_replay:
    replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
    if prioritized_replay_beta_iters is None: #Still quite a mistery to me
        prioritized_replay_beta_iters = max_timesteps
    beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                   initial_p=prioritized_replay_beta0,
                                   final_p=1.0)
else:
    replay_buffer = ReplayBuffer(buffer_size)
    beta_schedule = None


# Create the schedule for exploration starting from 1.
exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction *     max_timesteps),
                             initial_p=1.0,
                             final_p=exploration_final_eps)


#starts the session (at this point tensorflow allocates all the memory in your 
#GPU)
sess = U.make_session(num_cpu=num_cpu)
sess.__enter__()

# Initialize the parameters and copy them to the target network.
U.initialize()
update_target()

##############################
##      The Main Loop      ###
##############################
episode_rewards = [0.0]
saved_mean_reward = None
obs = env.reset()
reset = True
for t in range(max_timesteps):
    if callback is not None:
        if callback(locals(), globals()):
            break
    
    # update exploration epsilon or apply param_noise, if it is chosen
    kwargs = {}
    if not param_noise:
        update_eps = exploration.value(t)
        update_param_noise_threshold = 0.
    else:
        update_eps = 0.
        # Compute the threshold such that the KL divergence between perturbed and non-perturbed
        # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
        # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
        # for detailed explanation.
        update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
        kwargs['reset'] = reset
        kwargs['update_param_noise_threshold'] = update_param_noise_threshold
        kwargs['update_param_noise_scale'] = True
        
     
    
    #choose action and act
    action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
    reset = False
    new_obs, rew, done, _ = env.step(action)
    
    episode_rewards[-1] += rew
    
    # Store transition in the replay buffer.
    #if not a null action there is a small penalty   
    if action:
        rew += action_penalty
    replay_buffer.add(obs, action, rew, new_obs, float(done))
    obs = new_obs
    
    #checks if game has ended
    if done:
        obs = env.reset()
        episode_rewards.append(0.0)
        reset = True
        
    #training
    if t > learning_starts and t % train_freq == 0:
        #sample the experiences
        if prioritized_replay:
            experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
        else:
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None
        
        #actual training step
        td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
        
        #Update priorities if prioritized replay on
        if prioritized_replay:
            new_priorities = np.abs(td_errors) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)
    
    
    #update target network
    if t > learning_starts and t % target_network_update_freq == 0:
        # Update target network periodically.
        update_target()
    
    #keep stats on progress
    num_episodes = len(episode_rewards)
    if num_episodes > 1:
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
    else:    
        mean_100ep_reward = episode_rewards[0]

    #print every once in a while
    if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
        print('steps', t, 'episodes', num_episodes,'reward', episode_rewards[-2], 'mean 100', mean_100ep_reward, 'e-value %.2f' %
              (exploration.value(t)))
    
    #save models
    if (checkpoint_freq is not None and t > learning_starts and
        num_episodes > 100 and t % checkpoint_freq == 0):
        if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
            if print_freq is not None:
                print('Saving model...')
            U.save_state(model_file)
            model_saved = True
            saved_mean_reward = mean_100ep_reward
            with open(rewards_file, 'wb') as f:
                pickle.dump(episode_rewards, f)
print('done, did %s steps' % (t+1))
