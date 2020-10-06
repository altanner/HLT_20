import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as plticker


def plot_config():

    """ Set up plot visual params for Kalman Filter """

    mpl.use("Agg")

    plt.style.use(["seaborn-whitegrid", "seaborn-ticks"])

    rcParams["figure.figsize"] = 12, 8
    rcParams["axes.facecolor"] = "FFFFFF"
    rcParams["savefig.facecolor"] = "FFFFFF"
    rcParams["figure.facecolor"] = "FFFFFF"

    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"

    rcParams["mathtext.fontset"] = "cm"
    rcParams["mathtext.rm"] = "serif"

    rcParams.update({"figure.autolayout": True})

