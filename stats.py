import pickle
import cv2


class Stats:
    def __init__(self, stats_file):
        self.stats_file = stats_file
        
        self.life_episode = {
        'clipped reward': [0.0],
        'real reward': [0.0],
        'non null action': [0.0],
        'sizes': [0],
        'frames': [],
        'actions': [],
        'q-values': [],
        'rewards': [],
        }
        
        self.real_episode = {
        'clipped reward': [0.0],
        'real reward': [0.0],
        'non null action': [0.0],
        'sizes': [0],
        'frames': [],
        'actions': [],
        'q-values': [],
        'rewards': [],
        }
    
    def update_stats(self, frame, action, q_values, rew, real_reward):
        
        #update episode sizes
        self.life_episode['sizes'][-1] += 1
        self.real_episode['sizes'][-1] += 1
        
        #update frame
        self.life_episode['frames'].append(frame)
        self.real_episode['frames'].append(frame)
        
        #actions
        self.life_episode['actions'].append(action)
        self.real_episode['actions'].append(action)
        
        #q-values
        self.life_episode['q-values'].append(q_values)
        self.real_episode['q-values'].append(q_values)
        
        #rewards
        self.life_episode['rewards'].append(rew)
        self.real_episode['rewards'].append(rew)

        if rew:
            #update clipped rewards
            self.life_episode['clipped reward'][-1] += rew
            self.real_episode['clipped reward'][-1] += rew
            #Update real rewards 
            self.life_episode['real reward'][-1] += real_reward
            self.real_episode['real reward'][-1] += real_reward
        
        #action stats
        if action:
            #Update non null actions
            self.life_episode['non null action'][-1] += 1
            self.real_episode['non null action'][-1] += 1
    
    def save_stats(self):
        stats = {
        'life episode': self.life_episode,
        'real episode': self.real_episode,
        }
        with open(self.stats_file,'wb') as f:
            pickle.dump(stats, f)
    
    def new_life_episode(self):
        self.life_episode['non null action'][-1] /= self.life_episode['sizes'][-1]
        
        self.life_episode['sizes'].append(0)
        self.life_episode['frames'] = []
        self.life_episode['actions'] = []
        self.life_episode['q-values'] = []
        self.life_episode['rewards'] = []
        self.life_episode['clipped reward'].append(0.0)
        self.life_episode['real reward'].append(0.0)
        self.life_episode['non null action'].append(0.0)
    
    def new_real_episode(self):
        self.real_episode['non null action'][-1] /= self.real_episode['sizes'][-1]
        self.real_episode['sizes'].append(0)
        self.real_episode['frames'] = []
        self.real_episode['actions'] = []
        self.real_episode['q-values'] = []
        self.real_episode['rewards'] = []
        self.real_episode['clipped reward'].append(0.0)
        self.real_episode['real reward'].append(0.0)
        self.real_episode['non null action'].append(0.0)
    
    def save_video(self, vid_file):
        frames = self.real_episode['frames']
        height, width, layers = frames[0].shape
    
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video = cv2.VideoWriter(vid_file, fourcc, 30, (width,height))
    
        for frame in frames:
            r,g,b = cv2.split(frame)
            bgr_img = cv2.merge([b,g,r])
            video.write(bgr_img)
        video.release()
        
        #additional info from the current video
        info = {
        'actions': self.real_episode['actions'],
        'q-values': self.real_episode['q-values'],
        'rewards': self.real_episode['rewards'],
        }
        with open(vid_file+'.info','wb') as f:
            pickle.dump(info,f)
