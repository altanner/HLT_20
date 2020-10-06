import time
import numpy as np
import argparse
from pprint import pprint
import tensorflow as tf
from genKFTracks2d import gen_tracks

np.random.seed(42)

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", type=int, dest="n", default=1, help="nInputs")
args = argParser.parse_args()
n_gen = args.n


plane_distance = 1.0  # Distance between planes
sigma = 10e-2         # Resolution of planes #! why is this and sigma diff from gentracks?
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

#! C0 is not mentioned in the paper (?)
C0 = np.array([[sigma ** 2, 0, 0, 0],
               [0, np.pi, 0, 0],
               [0, 0, sigma ** 2, 0],
               [0, 0, 0, np.pi]])

F_1 = tf.constant(F, dtype=tf.float32)
F_scalar = tf.constant(F_1, dtype=tf.float32)

G = tf.constant(G, dtype=tf.float32)
H = tf.constant(H, dtype=tf.float32)
Q = tf.constant(Q, dtype=tf.float32)
C0 = tf.constant(C0, dtype=tf.float32)

projectedTrack = None
projectedCov = None

filteredTrack = None
filteredCov = None

F_init = tf.Variable(np.tile(F_1, (n_gen, 1, 1)), dtype=tf.float32)
F = tf.Variable(np.tile(F_1, (n_gen, 1, 1)), dtype=tf.float32)


def residual(hits, p_filtered, H):

    # Pad to shape of p, transpose to col vector
    hits_full_dim = tf.transpose(tf.pad(tf.expand_dims(hits, 1), [[0, 0], [0, 1]]))

    return hits_full_dim - (H @ tf.transpose(p_filtered))


def chiSquared(residual, G, C_proj, p_proj, p_filt):

    t1 = tf.einsum("iB,jB -> B", residual, G @ residual)

    p_diff = p_filt - p_proj

    C_diff = tf.einsum("Bij,Bj->Bi", tf.linalg.inv(C_proj), p_diff)

    t2 = tf.einsum("Bi,Bj -> B", p_diff, C_diff)

    return t1 + t2


def project(F, p, C, Q):

    # p_proj = tf.einsum('ji,iB->Bj', F_scalar, p)

    # With vector of Fs
    p_proj = tf.einsum("Bji,iB->Bj", F, p)

    C_proj = tf.transpose(F_scalar @ C) @ tf.transpose(F_scalar) + Q

    return p_proj, C_proj


def filter(p_proj, C_proj, H, G, m):

    HG = tf.transpose(H) @ G

    # Innermost two axies must be 'matrix'
    inv_C_proj = tf.linalg.inv(C_proj)

    C = tf.linalg.inv(inv_C_proj + HG @ H)

    # Reversing batch dimension -> fix me!
    p = tf.einsum("Bij,Bj->Bi", inv_C_proj, p_proj) + tf.einsum("ji,iB->Bj", HG, m)
    p = tf.einsum("Bij,Bj->Bi", C, p)

    return p, C


def bkgTransport(C, F, C_proj):

    #  Extra transpose (both) to make this work with axis ordering

    return C @ tf.transpose(F, (0, 2, 1)) @ tf.linalg.inv(C_proj)


def smooth(p_k1_smooth, p_k1_proj, C_k1_smooth, C_k1_proj, p_filtered, C_filtered, A):

    # Also reversed batches!
    p_smooth = p_filtered + tf.einsum("Bij,jB->iB", A, p_k1_smooth - p_k1_proj)

    # Transpose only inner 'matrix' dimensions
    C_smooth = C_filtered + A @ (C_k1_smooth - C_k1_proj) @ tf.transpose(A, (0, 2, 1))

    return p_smooth, C_smooth


def project_and_filter_internal(
    i, m, hits, p, C, filteredTrack, filteredCov, projectedTrack, projectedCov
):

    global F #! ?

    p = filteredTrack[i - 1]
    C = filteredCov[i - 1]

    p_proj, C_proj = project(F, p, C, Q)

    m[0, :].assign(hits[:, i, 0])
    m[2, :].assign(hits[:, i, 1])

    p_filt, C_filt = filter(p_proj, C_proj, H, G, m)

    # res = residual(hits[:,i], p_filt, H)
    # chiSq = chiSquared(res, G, C_proj, p_proj, p_filt)

    # skipIdxs = tf.where(chiSq > 100. * tf.ones(chiSq.shape))

    # p_proj = tf.tensor_scatter_nd_update(p_proj, skipIdxs, tf.squeeze(tf.gather(projectedTrack[i-1], skipIdxs), axis = 1))
    # C_proj = tf.tensor_scatter_nd_update(C_proj, skipIdxs, tf.squeeze(tf.gather(projectedCov[i-1], skipIdxs), axis = 1))
    #
    # p_filt = tf.tensor_scatter_nd_update(p_filt, skipIdxs, tf.squeeze(tf.gather(tf.transpose(filteredTrack[i-1]), skipIdxs), axis = 1))
    # C_filt = tf.tensor_scatter_nd_update(C_filt, skipIdxs, tf.squeeze(tf.gather(tf.transpose(filteredCov[i-1], (2, 0, 1)), skipIdxs), axis = 1))
    #
    # # Reset, in case we set this to + 1 last time
    # F = F_init
    # F = tf.tensor_scatter_nd_update(F, skipIdxs, tf.squeeze(tf.gather(F, skipIdxs), axis = 1) + updF)

    # TODO: Sort out this transpose nightmare....
    p_filt = tf.transpose(p_filt)
    C_filt = tf.transpose(C_filt, (1, 2, 0))

    return p_proj, C_proj, p_filt, C_filt


if __name__ == "__main__":

    # n_gen defined globally #!
    #! input are generated tracks.
    #! outputs are project track, proj covariance,
    #!             smooth track, smooth covariance,
    #!             filtered track, filtered covariance
    hits, trueTracks = gen_tracks(n_gen=n_gen)

    hits = tf.constant(hits, dtype=tf.float32)

    m0 = tf.Variable(tf.zeros((4, n_gen)))  # (hit_x, slope_x, hit_y, slope_y)

    m0[0, :].assign(hits[:, 0, 0])  # First plane, x hits
    m0[2, :].assign(hits[:, 0, 1])  # First plane, y hits

    p0 = m0

    C0 = tf.constant(np.stack([C0 for i in range(n_gen)], -1), dtype=tf.float32)

    start = time.perf_counter()

    p_proj, C_proj = project(F, p0, C0, Q)

    p, C = filter(p_proj, C_proj, H, G, m0)

    p = tf.transpose(p)
    C = tf.transpose(C, (1, 2, 0))

    projectedTrack = tf.Variable([p_proj for i in range(plane_count)])
    projectedCov = tf.Variable([C_proj for i in range(plane_count)])

    filteredTrack = tf.Variable([p for i in range(plane_count)])
    filteredCov = tf.Variable([C for i in range(plane_count)])

    m = tf.Variable(tf.zeros((4, n_gen)))

    for i in range(1, plane_count):

        #! project forward, filter /smooth backwards.
        p_proj, C_proj, p_filt, C_filt = project_and_filter_internal(
            tf.constant(i),
            m,
            hits,
            p,
            C,
            filteredTrack,
            filteredCov,
            projectedTrack,
            projectedCov)

        filteredTrack[i].assign(p_filt)
        filteredCov[i].assign(C_filt)

        projectedTrack[i].assign(p_proj)
        projectedCov[i].assign(C_proj)

    smoothedTrack = tf.Variable([filteredTrack[-1] for i in range(plane_count)])
    smoothedCov = tf.Variable([tf.transpose(filteredCov[-1]) for i in range(plane_count)])

    reversedPlaneIndices = list(range(0, plane_count - 1))
    reversedPlaneIndices.reverse()

    for i in reversedPlaneIndices:

        p_k1_proj, C_k1_proj = projectedTrack[i + 1], projectedCov[i + 1]
        p_filtered, C_filtered = filteredTrack[i], filteredCov[i]
        p_k1_smooth, C_k1_smooth = smoothedTrack[i + 1], smoothedCov[i + 1]

        A = bkgTransport(tf.transpose(C_filtered, (2, 0, 1)), F, C_k1_proj)

        p_smooth, C_smooth = smooth(
            p_k1_smooth,
            tf.transpose(p_k1_proj),
            C_k1_smooth,
            C_k1_proj,
            p_filtered,
            tf.transpose(C_filtered, (2, 0, 1)),
            A)

        smoothedTrack[i].assign(p_smooth)
        smoothedCov[i].assign(C_smooth)

    end = time.perf_counter()

    print(f"{end - start}")

   # print(f"Projected\n {p_proj} \n Filtered\n {p_filt} \n Smooth\n {p_smooth}")
