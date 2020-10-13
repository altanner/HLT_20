import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


np.random.seed(42)

plane_distance = 1.0  # d = Distance between planes
sigma = 15e-2         # sigma = Resolution of planes
plane_count = 5       # N = Number of planes
z = 0.1               # z = Thickness of absorber #~ unused
x0 = 0.01             # x0 = Radiation length of absorber #~ unused
theta0 = 15e-3        # theta0 = Multiple scattering uncertainty
#(TODO: use formula) < legacy #? <<<<<<<<

initial_theta_range = [-np.arcsin(1 / 5.0),
                       np.arcsin(1 / 5.0)]

initial_phi_range = initial_theta_range
#initial_phi_range = [-np.arcsin(1 / 5.0), #~ will phi always = theta?
 #                  np.arcsin(1 / 5.0)]

x_range = [plane_count * plane_distance * np.tan(initial_theta_range[0]),
           plane_count * plane_distance * np.tan(initial_theta_range[1])]

y_range = x_range #~ will y_range always = x_range?
#y_range = [plane_count * plane_distance * np.tan(initial_phi_range[0]),
 #         plane_count * plane_distance * np.tan(initial_phi_range[1])]

def plot_hits(plane_count, plot_tracks, digi_hits, plane_range, name, ax=None):

    """
    #todo docstring
    """

    for i in range(1, plane_count + 1):
        ax.plot([i, i], [-3, 3], color="k", linestyle="--", alpha=0.15)

    ax.plot(np.array(range(0, plane_count + 1)), plot_tracks.T)
    ax.plot(np.array(range(1, plane_count + 1)), digi_hits.T, "x", color="k")

    ax.set_ylim(plane_range[0] - 0.1, plane_range[1] + 0.1)
    ax.set_xlim(-0.25, plane_count + 0.25)


def plot_hits2d(plane_count, plot_tracks, digi_hits, plane_range_x, plane_range_y, ax=None):

    """
    #todo docstring
    """

    # Plot this first as a hack to get the track colour consistent between plots

    ax.plot(plot_tracks[:, :, 0].T, plot_tracks[:, :, 1].T)

    for p in range(plane_count):

        # First plane
        x_hits = digi_hits[:, p, 0]
        y_hits = digi_hits[:, p, 1]

        ax.plot(x_hits, y_hits, "x")
        ax.set_xlim(plane_range_x[0] - 0.1, plane_range_x[1] + 0.1)
        ax.set_ylim(plane_range_y[0] - 0.1, plane_range_y[1] + 0.1)


def dist(h1, h2): #~ <<< distance of what?

    return np.linalg.norm(h1 - h2)


def exchange_track_hits(reco_hits, frac=0.2, prob=0.2):

    new_hits = reco_hits.copy()

    #~ need to understand this.
    #~
    # Exchange closest frac hits with probability prob
    # Only exchange hits on the same plane
    # Want to avoid exchanging hits for same tracks?

    for plane in range(plane_count):

        plane_hits = new_hits[:, plane, :]

        k = int(len(plane_hits) * frac)
        select = int(len(plane_hits) * frac * prob)

        # Do the naive thing first, revisit if it's too slow

        dists = np.array(
            [
                [
                    (dist(h1, h2) if not np.all(np.equal(h1, h2)) else np.inf)
                    for h1 in plane_hits
                ]
                for h2 in plane_hits
            ]
        )

        # Choose a minimal distance assignment that corresponds to the swaps

        s = linear_sum_assignment(dists)
        s = list(zip(s[0], s[1]))

        # Pick the k nearest hits

        s = sorted(s, key=lambda p: dists[p])
        s = np.array(s[:k])

        # Choose select at random

        sIdx = np.random.choice(range(len(s)), select, replace=False)
        s = list(map(tuple, s[sIdx]))

        # Do the exchange

        plane_hits[[x[0] for x in s], :], plane_hits[[x[1] for x in s], :] = (
            plane_hits[[x[1] for x in s], :],
            plane_hits[[x[0] for x in s], :],
        )

    return new_hits


def gen_tracks(n_gen=10, truthOnly=False, plot=False, exchange_hits=False):

    # Absorber lengths add #~ unused as yet
    ms_dists = np.array([i * plane_distance for i in range(1, plane_count + 1)])

    # But resulting MS uncertainties add in quadrature
    # TODO: Correct for the actual path length (more oblique tracks see more material)
    #~ so is ms_dists included?
    scatter_errors = np.array([np.sqrt(i) * theta0 for i in range(1, plane_count + 1)])

    #~ resolution (sigma) to define bins
    x_bins = np.arange(x_range[0] - 2 * sigma, x_range[1] + 2 * sigma, sigma)
    y_bins = np.arange(y_range[0] - 2 * sigma, y_range[1] + 2 * sigma, sigma)

    tan_thetas = np.tan(np.random.uniform(*initial_theta_range, n_gen))
    tan_phis = np.tan(np.random.uniform(*initial_phi_range, n_gen))

    # tan_thetas = np.tan([0 for i in range(n_gen)])
    x_true_hits = np.outer(tan_thetas,
                           plane_distance * np.array(range(1, plane_count + 1)))
    y_true_hits = np.outer(tan_phis,
                           plane_distance * np.array(range(1, plane_count + 1)))

    #~ np.stack joins arrays along new axis
    true_hits = np.stack((x_true_hits, y_true_hits), -1)

    x_plot_tracks = np.hstack((np.zeros((n_gen, 1)),
                               x_true_hits))  # Project tracks back to origin @ -1

    y_plot_tracks = np.hstack((np.zeros((n_gen, 1)),
                               y_true_hits))  # Project tracks back to origin @ -1

    plot_tracks = np.stack((x_plot_tracks, y_plot_tracks), -1)

    # Fix me #? what needs fixing?
    msGauss = np.random.normal(np.zeros(plane_count),
                               scatter_errors,
                               (n_gen, plane_count))

    x_hits = x_true_hits + msGauss #~ apply scatter
    y_hits = y_true_hits + msGauss #~ apply scatter

    #~ create discrete bins of x hits
    x_hit_map = np.digitize(x_hits, x_bins)
    x_digi_hits = x_bins[x_hit_map]

    #~ create discrete bins of y hits
    y_hit_map = np.digitize(y_hits, y_bins)
    y_digi_hits = y_bins[y_hit_map]

    #~ combine both x and y hits into full hit map
    digi_hits = np.stack((x_digi_hits, y_digi_hits), -1)

    if exchange_hits: #~ what is exchange hits?
        digi_hits = exchange_track_hits(digi_hits, frac=0.35, prob=0.75)

    if plot:

        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(2, 2, 1)
        plot_hits(plane_count, x_plot_tracks, x_digi_hits, x_range, "x", ax=ax)

        ax = fig.add_subplot(2, 2, 2)
        plot_hits(plane_count, y_plot_tracks, y_digi_hits, y_range, "y", ax=ax)

        ax = fig.add_subplot(2, 2, 3)
        plot_hits2d(plane_count, plot_tracks, digi_hits, x_range, y_range, ax=ax)

        plt.savefig("gen_tracks.pdf")
        plt.clf()

    if truthOnly:
        return true_hits
    else:
        return digi_hits, plot_tracks


if __name__ == "__main__":

#    plot_config() #~ tidy up plot calls
    gen_tracks(n_gen=1, plot=True) #~ make pdf plot
    #~ (when not main, gentracks is called by tf_kalman)
    print(gen_tracks(n_gen=1))
