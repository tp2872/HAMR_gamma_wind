''''
In this function I am defining two sets of matplotlib parameters for paper or presentation version of figures, where applicable. 
I intend to important this into my jupyter notebooks so their matplotlib settings will be synchronized.
'''

import matplotlib.pyplot as plt

# For use in paper
def paper_params():
    plt.rcParams['figure.figsize'] = (10,8)
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['font.family'] = 'stixgeneral'
    plt.rcParams['font.size'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['axes.labelsize'] = 18

    plt.style.use("default")
    return

# For use in presentation
def presentation_params():
    plt.rcParams['figure.figsize'] = (10,8)
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['font.family'] = 'stixgeneral'
    plt.rcParams['font.size'] = 24
    plt.rcParams['xtick.labelsize'] = 26
    plt.rcParams['ytick.labelsize'] = 26
    plt.rcParams['axes.labelsize'] = 34

    plt.style.use("dark_background")
    return
