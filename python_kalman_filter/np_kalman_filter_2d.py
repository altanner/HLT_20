import time
import numpy as np
from numpy.linalg import inv
from pprint import pprint  #! unused
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as plticker
from genKFTracks2d import gen_tracks, x_range, y_range

np.random.seed(42)


plane_distance = 1.0  # Distance between planes
sigma = 10e-2         # Resolution of planes
plane_count = 5       # Number of planes
z = 0.1               # Thickness of absorber
x0 = 0.01             # Radiation length of absorber
theta0 = 10e-3        # Multiple scattering uncertainty (TODO: use formula)

#! initiate the matrices
#! F is the transfer matrix
F = np.array([[1, plane_distance, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, plane_distance],
              [0, 0, 0, 1]])

#! G is the noise matrix
G = np.array([[1 / sigma ** 2, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1 / sigma ** 2, 0],
              [0, 0, 0, 0]])

#! H the relation between the measurement m and the state p
H = np.array([[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 0]])

#! Q is the random error matrix, ie the scatter
Q = np.zeros(4)

#! C0 is the initial parameters
C0 = np.array([[sigma ** 2, 0, 0, 0],
               [0, np.pi, 0, 0],
               [0, 0, sigma ** 2, 0],
               [0, 0, 0, np.pi]])


def plotHits(plane_count, plotTracks, digiHits,
             planeRange, name, ax=None):

    """#todo docstring
    all of this can be refactored
    into a plot / vis module"""

    for i in range(1, plane_count + 1):
        ax.plot([i, i], [-3, 3], color="k", linestyle="--", alpha=0.15)

    ax.plot(np.array(range(1, plane_count + 1)), plotTracks.T, alpha=0.75)
    ax.plot(np.array(range(1, plane_count + 1)), digiHits, "x", color="k")

    ax.set_ylim(planeRange[0] - 0.1, planeRange[1] + 0.1)
    ax.set_xlim(-0.25, plane_count + 0.25)

    ax.text(0.25, 0.80, "$" + name + "$", fontsize=18)

    ax.set_ylabel(r"$x$" if name == "y" else r"$y$", fontsize=16)
    ax.set_xlabel(r"$z$", fontsize=16)

    # this locator puts ticks at regular intervals
    loc = plticker.MultipleLocator(base=0.5)
    ax.yaxis.set_major_locator(loc)


def plotHits2d(plane_count, plotTracks, digiHits,
               planeRangeX, planeRangeY, ax=None):

    """#todo docstring
    all of this can be refactored
    into a plot / vis module"""

    # Plot this first as a hack to get the track colour consistent between plots
    ax.plot(smoothedTrack[:, 0, :], smoothedTrack[:, 2, :], lw=0.5)

    for plane in range(plane_count):

        # First plane
        xHits = digiHits.T[0, plane, :]
        yHits = digiHits.T[1, plane, :]

        ax.plot(xHits, yHits, "x", alpha=0.2)
        ax.set_xlim(planeRangeX[0] - 0.1, planeRangeX[1] + 0.1)
        ax.set_ylim(planeRangeY[0] - 0.1, planeRangeY[1] + 0.1)

    ax.text(-0.90, 0.80, r"$z$", fontsize=18)

    ax.set_ylabel(r"$x$", fontsize=16)
    ax.set_xlabel(r"$y$", fontsize=16)

    loc = plticker.MultipleLocator(base=0.5)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)


def propagateStateEKF(x):

    #! what is this?
    x_new = x.copy()
    x_new[0, :] = x[0, :] + plane_distance * np.tan(x[1, :])
    x_new[2, :] = x[2, :] + plane_distance * np.tan(x[3, :])

    return x_new.T


def jacobianF(x):

    #! not mentioned in paper
    #! what is this?
    # No batch 'eye' << wtf
    f = np.zeros((x.shape[1], x.shape[0], x.shape[0]))

    f[:, 0, 0] = 1
    f[:, 1, 1] = 1
    f[:, 2, 2] = 1
    f[:, 3, 3] = 1

    f[:, 0, 1] = 1.0 / (np.cos(x[1, :]) + 1e-10) ** 2
    f[:, 2, 3] = 1.0 / (np.cos(x[3, :]) + 1e-10) ** 2

    return f


#! func taking transfer matrix F, "p", "C" and random error matrix Q
def projectEKF(F, p, C, Q):

    p_proj = propagateStateEKF(p)

    jF = jacobianF(p)
    C_proj = (jF @ C.T) @ np.transpose(jF, (0, 2, 1)) + Q

    return p_proj, C_proj


#! diff project <> projectEKF?
def project(F, p, C, Q):

    p_proj = np.einsum("ji,iB->Bj", F, p)

    C_proj = (F @ C).T @ F.T + Q

    return p_proj, C_proj


def filter(p_proj, C_proj, H, G, m):

    HG = H.T @ G

    # Innermost two axes must be 'matrix'
    inv_C_proj = inv(C_proj)

    C = inv(inv_C_proj + HG @ H)

    p = np.einsum("Bij,Bj->Bi", inv_C_proj, p_proj) + np.einsum("ji,iB->Bj", HG, m)
    p = np.einsum("Bij,Bj->Bi", C, p)

    return p, C


#! backward transport (?)
def bkgTransport(C, F, C_proj):

    #  Extra transpose (both) to make this work with axis ordering

    return C @ F.T @ inv(C_proj)


def smooth(p_k1_smooth, p_k1_proj, C_k1_smooth, C_k1_proj, p_filtered, C_filtered, A):

    # Also reversed batches!
    p_smooth = p_filtered + np.einsum("Bij,jB->iB", A, p_k1_smooth - p_k1_proj)

    # Transpose only inner 'matrix' dimensions
    C_smooth = C_filtered + A @ (C_k1_smooth - C_k1_proj) @ np.transpose(A, (0, 2, 1))

    return p_smooth, C_smooth


def residual(hits, p_filtered, H):

    return hits - (H @ p_filtered)


def chiSquared(residual, G, C_proj, p_proj, p_filt):

    t1 = residual.T @ G @ residual

    p_diff = p_filt - p_proj
    t2 = p_diff.T @ inv(C_proj) @ p_diff

    return t1 + t2


if __name__ == "__main__":

    #! main is too long. refactor out.

    start = time.perf_counter()

    n_gen = 7 #! send to args
    hits, trueTracks = gen_tracks(n_gen=n_gen)

    m0 = np.zeros((4, n_gen))
    m0[0, :] = hits[:, 0, 0]  # First plane, x hits
    m0[2, :] = hits[:, 0, 1]  # First plane, y hits

    p0 = m0

    C0 = np.stack([C0 for i in range(n_gen)], -1)


    # Batch dim second for p #! ?
    # p_proj, C_proj = projectEKF(F, p0, C0, Q)
    p_proj, C_proj = project(F, p0, C0, Q)

    p, C = filter(p_proj, C_proj, H, G, m0)

    # Because batch dims are inconsistent...
    p = p.T
    C = np.transpose(C, (1, 2, 0))

    #! forward projection
    projectedTrack = [p_proj]
    projectedCov = [C_proj]

    filteredTrack = [p]
    filteredCov = [C]

    for i in range(1, plane_count):
        # p_proj, C_proj = projectEKF(F, p, C, Q)
        p_proj, C_proj = project(F, p, C, Q)

        m = np.zeros((4, n_gen))
        m[0, :] = hits[:, i, 0]  # ith plane, x hits
        m[2, :] = hits[:, i, 1]  # ith plane, y hits

        p, C = filter(p_proj, C_proj, H, G, m)

        p = p.T
        C = np.transpose(C, (1, 2, 0))

        filteredTrack.append(p)
        filteredCov.append(C)

        projectedTrack.append(p_proj)
        projectedCov.append(C_proj)

    #! this looks weird
    smoothedTrack = [None for i in range(plane_count - 1)] + [filteredTrack[-1]]
    smoothedCov = [None for i in range(plane_count - 1)] + [filteredCov[-1]]

    reversedPlaneIndices = list(range(0, plane_count - 1))
    reversedPlaneIndices.reverse() #! combine lines / simplify

    for i in reversedPlaneIndices:
        #! backward smoothing [3]
        p_k1_proj, C_k1_proj = projectedTrack[i + 1], projectedCov[i + 1]
        p_filtered, C_filtered = filteredTrack[i], filteredCov[i]
        p_k1_smooth, C_k1_smooth = smoothedTrack[i + 1], smoothedCov[i + 1]

        if i == reversedPlaneIndices[0]:
            C_k1_smooth = np.transpose(C_k1_smooth, (2, 0, 1))

        # Need to have 7, 2, 2 shape because of inversion - fix me!! #! ?
        A = bkgTransport(np.transpose(C_filtered, (2, 0, 1)), F, C_k1_proj)

        p_smooth, C_smooth = smooth(p_k1_smooth,
                                    p_k1_proj.T,
                                    C_k1_smooth,
                                    C_k1_proj,
                                    p_filtered,
                                    np.transpose(C_filtered, (2, 0, 1)),
                                    A)

        smoothedTrack[i] = p_smooth
        smoothedCov[i] = C_smooth

    smoothedTrack = np.array(smoothedTrack)
    filteredTrack = np.array(filteredTrack)

    end = time.perf_counter() #! better way of timing?

    print("Elapsed time = {:.9f} seconds".format(end - start))

    #! refactor into vis / plot >> args
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(2, 2, 1)
    plotHits(plane_count,
             smoothedTrack[:, 0, :].T,
             hits.T[0, :, :],
             x_range,
             "x",
             ax=ax)

    ax = fig.add_subplot(2, 2, 2)
    plotHits(plane_count,
             smoothedTrack[:, 2, :].T,
             hits.T[1, :, :],
             x_range,
             "y",
             ax=ax)

    ax = fig.add_subplot(2, 2, 3)
    plotHits2d(plane_count,
               smoothedTrack,
               hits,
               x_range,
               y_range,
               ax=ax)

    plt.savefig("kfTracks.pdf")
    plt.clf()
