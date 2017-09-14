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
from atari_wrappers import wrap_dqn, ScaledFloatFrame, SaveCurrents

#I should move this funciton somewhere else once I get the chance
import cv2  
def save_video(frames, file):
    height, width, layers = frames[0].shape
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(file, fourcc, 30, (width,height))
    
    for frame in frames:
        r,g,b = cv2.split(frame)
        bgr_img = cv2.merge([b,g,r])
        video.write(bgr_img)
    video.release()

#Defines the folder where the results will be saved
folder = 'New_stats/'

#Number of the expirenment
expirement = 'Video_test1'

#name of envirenment
environment = 'Seaquest'
env_name = environment+'NoFrameskip-v4'

#Builds the file name for where the model and stats will be stored
name = environment+'_'+expirement
model_name = name+'_model'
rewards_name = name+'_rewards.pkl'
stats_name = name+'_stats.pkl'
description_name = name+'.description'
model_file = os.path.join(folder, model_name)
rewards_file = os.path.join(folder, rewards_name)
stats_file = os.path.join(folder, stats_name)
description_file = os.path.join(folder, description_name)
model_saved = False


#Action penalty parameter
action_penalty = 0.01

#Objective
#A description text for future reference
objective = 'Testing the new stats gathering mechanisms'

#Parameters of Q-learning that could potentialy be tweaked
lr=1e-4
max_timesteps=5000000
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
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=None
prioritized_replay_eps=1e-6
num_cpu=16
param_noise=False
callback=None

#Save/Show Frequencies
print_freq=2
checkpoint_freq=50000
save_stats = 50000
save_video_freq = 500000


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
input()
with open(description_file, 'w') as f:
    f.write(description)


#makes the environment, and wraps it with a couple of preprocessing layers.
orig_env = gym.make(env_name)
orig_env = SaveCurrents(orig_env) #keeps original env accessible
env = ScaledFloatFrame(wrap_dqn(orig_env))


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

#inicialize the stats to be collected
episode_cliped_rew = [0.0]
#Episodic Life stats
life_episode = {
'clipped reward': [0.0],
'real reward': [0.0],
'non null action': [0.0],
'frames': [],
'sizes': [0],
}

#Real episode stats
real_episode = {
'clipped reward': [0.0],
'real reward': [0.0],
'non null action': [0.0],
'frames': [],
'sizes': [0],
}

saved_mean_reward = None
obs = env.reset()
reset = True
video = False
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
    
    
    #############
    ####Stats####
    #############
    
    #Update size
    life_episode['sizes'][-1] += 1
    real_episode['sizes'][-1] += 1
    #Update episode observations
    life_episode['frames'].append(orig_env.current_obs)
    real_episode['frames'].append(orig_env.current_obs)
    #reward stats
    if rew:
        #update clipped rewards
        life_episode['clipped reward'][-1] += rew
        real_episode['clipped reward'][-1] += rew
        episode_cliped_rew[-1] += rew
        #Update real rewards
        real_reward = sum(orig_env.last_4_rews) 
        life_episode['real reward'][-1] += real_reward
        real_episode['real reward'][-1] += real_reward
    #action stats
    if action:
        #Update non null actions
        life_episode['non null action'][-1] += 1
        real_episode['non null action'][-1] += 1
    
    
        
    # Store transition in the replay buffer.
    #if not a null action there is a small penalty   
    if action:
        rew += action_penalty
    replay_buffer.add(obs, action, rew, new_obs, float(done))
    obs = new_obs
        
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
    num_episodes = len(episode_cliped_rew)
    if num_episodes > 1:
        mean_100ep_reward = round(np.mean(episode_cliped_rew[-101:-1]), 1)
    else:    
        mean_100ep_reward = episode_cliped_rew[0]

    
    #save models
    if (checkpoint_freq is not None and t > learning_starts and
        num_episodes > 100 and t % checkpoint_freq == 0):
        if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
            if print_freq is not None:
                print('Saving model...')
            U.save_state(model_file)
            model_saved = True
            saved_mean_reward = mean_100ep_reward
    #save stats
    if t % save_stats == 0: 
        stats = {
        'life episode': life_episode,
        'real episode': real_episode,
        }
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        with open(rewards_file, 'wb') as f:
            pickle.dump(episode_cliped_rew, f)
    
    #save video
    if t % save_video_freq == 0:
        #the actual video saving is done at the end of the episode,
        #so here we just set a flag.
        video = True
    
    
    #Life episode end
    if done:
        #Real end of Episode
        if orig_env.current_done:
            if video:
                vid_file = os.path.join(folder, name+str(t)+'.avi')
                save_video(real_episode['frames'],vid_file)
                video = False
            #Reset variables associated with real episode
            real_episode['non null action'][-1] /= real_episode['sizes'][-1]
            real_episode['sizes'].append(0)
            real_episode['frames'] = []
            real_episode['clipped reward'].append(0.0)
            real_episode['real reward'].append(0.0)
            real_episode['non null action'].append(0.0)
            
                
        #Resets variables associated with life episode
        episode_cliped_rew.append(0.0)
        life_episode['non null action'][-1] /= life_episode['sizes'][-1]
        
        life_episode['sizes'].append(0)
        life_episode['frames'] = []
        life_episode['clipped reward'].append(0.0)
        life_episode['real reward'].append(0.0)
        life_episode['non null action'].append(0.0)
        reset = True

        obs = env.reset()
        #print every once in a while
        if print_freq is not None and len(episode_cliped_rew) % print_freq == 0:
            print('steps', t, 
                  'episodes', num_episodes,
                  'reward', episode_cliped_rew[-2], 
                  'mean 100', mean_100ep_reward, 
                  'e-value %.2f' %
                    (exploration.value(t)))
            
print('done, did %s steps' % (t+1))
