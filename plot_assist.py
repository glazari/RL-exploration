from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

def maxi(array, window=2):
    n = window
    l = len(array)
    return [np.max(array[i:i+n]) for i in range(l-n+1)]

def mini(array, window=2):
    n = window
    l = len(array)
    return [np.min(array[i:i+n]) for i in range(l-n+1)]


def median(array, window=2):
    n = window
    l = len(array)
    return [np.median(array[i:i+n]) for i in range(l-n+1)]

def mean(array, window=2):
    n = window
    l = len(array)
    return [np.mean(array[i:i+n]) for i in range(l-n+1)]

def plot_compare(series=[],scale=[0,6000,0,120],titles=[],win=100):
    #Default colors
    colors = [
        ['blue','lightblue'],
        ['orange','yellow'],
        ['green','lightgreen'],
        ['red','pink'],
        ['purple','mediumorchid']
    ]
    
    n = len(series)
    assert len(titles) == n or not titles
    
    plt.figure(figsize=(20,5))
    for i in range(len(series)):
        color1, color2 = colors[i]
        
        plt.subplot(1,n,i+1)
        
        plt.plot(maxi(series[i],window=win),color=color2)
        plt.plot(mini(series[i],window=win),color=color2)
        plt.plot(median(series[i],window=win),color=color1)
        
        plt.axis(scale)
        
        if titles:
            plt.title(titles[i])

def display_videos(videos):
    n = len(videos)

    html_base = """
    <video width="190" height="210" controls>
        <source src="{%d}">
    </video>
    """

    html = ''.join([html_base % i for i in range(n)])
    return HTML(html.format(*videos))
