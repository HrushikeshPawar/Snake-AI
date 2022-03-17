import matplotlib.pyplot as plt
from IPython import display
import os


# Enable interactive mode
plt.ion()
PLOT_DIR = 'images'


def plot(scores, mean_scores, suffix, save_plot=False):

    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='red')
    plt.plot(mean_scores, color='green')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    if save_plot:

        if not os.path.exists(PLOT_DIR):
            os.mkdir(PLOT_DIR)

        IMG_Path = os.path.join(PLOT_DIR, f'Training History - ({suffix}).png')

        plt.savefig(IMG_Path)