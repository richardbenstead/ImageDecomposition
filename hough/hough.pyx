# NOTE: Minor changes to code from skimage

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: language_level=3
import numpy as np
cimport numpy as cnp

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport abs
from libc.math cimport fabs, sqrt, ceil, atan2, M_PI
from skimage.draw import circle_perimeter

cnp.import_array()

def _hough_line(cnp.ndarray img,
                cnp.ndarray[ndim=1, dtype=cnp.double_t] stheta,
                cnp.ndarray[ndim=1, dtype=cnp.double_t] ctheta):
    """Perform a straight line Hough transform.
    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of double
        Angles at which to compute the transform, in radians.
    Returns
    -------
    H : 2-D ndarray of uint64
        Hough transform accumulator.
    theta : ndarray
        Angles at which the transform was computed, in radians.
    distances : ndarray
        Distance values.
    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.
    """

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[ndim=2, dtype=cnp.int64_t] accum
    cdef cnp.ndarray[ndim=1, dtype=cnp.double_t] bins
    cdef Py_ssize_t max_distance, offset

    offset = <Py_ssize_t>ceil(sqrt(img.shape[0] * img.shape[0] +
                                   img.shape[1] * img.shape[1]))
    max_distance = 2 * offset + 1
    accum = np.zeros((max_distance, stheta.shape[0]), dtype=np.int64)
    count = np.ones((max_distance, stheta.shape[0]), dtype=np.int64)
    bins = np.linspace(-offset, offset, max_distance)

    # finally, run the transform
    cdef Py_ssize_t j, x, y, accum_idx
    cdef float sty

    for j in range(stheta.shape[0]):
        for y in range(img.shape[0]):
            sty = y * stheta[j]
            for x in range(img.shape[1]):
                accum_idx = round((ctheta[j] * x + sty)) + offset
                accum[accum_idx, j] += 128 - img[y, x]
                count[accum_idx, j] += 1 # (img[y,x] != 128) # ignore points that have already been hit

    return accum/count, bins


def _hough_circle(cnp.ndarray img,
                  cnp.ndarray[ndim=1, dtype=cnp.intp_t] radius,
                  char normalize=True, char full_output=False):
    """Perform a circular Hough transform.
    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    radius : ndarray
        Radii at which to compute the Hough transform.
    normalize : boolean, optional (default True)
        Normalize the accumulator with the number
        of pixels used to draw the radius.
    full_output : boolean, optional (default False)
        Extend the output size by twice the largest
        radius in order to detect centers outside the
        input picture.
    Returns
    -------
    H : 3D ndarray (radius index, (M + 2R, N + 2R) ndarray)
        Hough transform accumulator for each radius.
        R designates the larger radius if full_output is True.
        Otherwise, R = 0.
    """
    if img.ndim != 2:
        raise ValueError('The input image must be 2D.')

    cdef Py_ssize_t xmax = img.shape[0]
    cdef Py_ssize_t ymax = img.shape[1]

    # compute the nonzero indexes
    cdef cnp.ndarray[ndim=1, dtype=cnp.intp_t] x, y
    x, y = np.nonzero(img)

    cdef Py_ssize_t num_pixels = x.size

    cdef Py_ssize_t offset = 0
    if full_output:
        # Offset the image
        offset = radius.max()
        x = x + offset
        y = y + offset

    cdef Py_ssize_t i, p, c, num_circle_pixels, tx, ty
    cdef double incr
    cdef cnp.ndarray[ndim=1, dtype=cnp.intp_t] circle_x, circle_y

    cdef cnp.ndarray[ndim=3, dtype=cnp.double_t] acc = \
         np.zeros((radius.size,
                   img.shape[0] + 2 * offset,
                   img.shape[1] + 2 * offset), dtype=np.double)

    for i, rad in enumerate(radius):
        # Store in memory the circle of given radius
        # centered at (0,0)
        circle_x, circle_y = circle_perimeter(0, 0, rad)

        num_circle_pixels = circle_x.size

        with nogil:

            if normalize:
                incr = 1.0 / num_circle_pixels
            else:
                incr = 1

            # For each non zero pixel
            for p in range(num_pixels):
                # Plug the circle at (px, py),
                # its coordinates are (tx, ty)
                for c in range(num_circle_pixels):
                    tx = circle_x[c] + x[p]
                    ty = circle_y[c] + y[p]
                    if offset:
                        acc[i, tx, ty] += incr
                    elif 0 <= tx < xmax and 0 <= ty < ymax:
                        acc[i, tx, ty] += incr

    return acc


def _probabilistic_hough_line(cnp.ndarray img, Py_ssize_t threshold,
                              Py_ssize_t line_length, Py_ssize_t line_gap,
                              cnp.ndarray[ndim=1, dtype=cnp.double_t] theta,
                              seed=None):
    """Return lines from a progressive probabilistic line Hough transform.
    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int
        Threshold
    line_length : int
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : 1D ndarray, dtype=double
        Angles at which to compute the transform, in radians.
    seed : {None, int, `numpy.random.Generator`, optional}
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
        Seed to initialize the random number generator.
    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y0)),
      indicating line start and end.
    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """
    cdef Py_ssize_t height = img.shape[0]
    cdef Py_ssize_t width = img.shape[1]

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[ndim=2, dtype=cnp.uint8_t] mask = \
        np.zeros((height, width), dtype=np.uint8)
    cdef Py_ssize_t *line_end = \
        <Py_ssize_t *>PyMem_Malloc(4 * sizeof(Py_ssize_t))
    if not line_end:
        raise MemoryError('could not allocate line_end')
    cdef Py_ssize_t max_distance, offset, num_indexes, index
    cdef double a, b
    cdef Py_ssize_t nidxs, i, j, k, x, y, px, py, accum_idx, max_theta
    cdef Py_ssize_t xflag, x0, y0, dx0, dy0, dx, dy, gap, x1, y1, count
    cdef cnp.int64_t value, max_value,
    cdef int shift = 16
    cdef int good_line
    cdef Py_ssize_t nlines = 0
    cdef Py_ssize_t lines_max = 2 ** 15  # maximum line number cutoff
    cdef cnp.intp_t[:, :, ::1] lines = np.zeros((lines_max, 2, 2), dtype=np.intp)
    max_distance = 2 * <Py_ssize_t>ceil((sqrt(img.shape[0] * img.shape[0] +
                                              img.shape[1] * img.shape[1])))
    cdef cnp.int64_t[:, ::1] accum = np.zeros((max_distance, theta.shape[0]), dtype=np.int64)
    offset = max_distance / 2
    cdef Py_ssize_t nthetas = theta.shape[0]

    # compute sine and cosine of angles
    cdef cnp.double_t[::1] ctheta = np.cos(theta)
    cdef cnp.double_t[::1] stheta = np.sin(theta)

    # find the nonzero indexes
    cdef cnp.intp_t[:] y_idxs, x_idxs
    y_idxs, x_idxs = np.nonzero(img<128)

    # mask all non-zero indexes
    mask[y_idxs, x_idxs] = 1

    count = len(x_idxs)
    random_state = np.random.default_rng(seed)
    random_ = np.arange(count, dtype=np.intp)
    random_state.shuffle(random_)
    cdef cnp.intp_t[::1] random = random_

    while count > 0:
        count -= 1
        # select random non-zero point
        index = random[count]
        x = x_idxs[index]
        y = y_idxs[index]

        # if previously eliminated, skip
        if not mask[y, x]:
            continue

        value = 0
        max_value = threshold - 1
        max_theta = -1

        # apply hough transform on point
        for j in range(nthetas):
            accum_idx = round((ctheta[j] * x + stheta[j] * y)) + offset
            # accum[accum_idx, j] += 1
            accum[accum_idx, j] += img[y, x];
            value = accum[accum_idx, j]
            if value > max_value:
                max_value = value
                max_theta = j
        if max_value < threshold:
            continue

        # from the random point walk in opposite directions and find line
        # beginning and end
        a = -stheta[max_theta]
        b = ctheta[max_theta]
        x0 = x
        y0 = y
        # calculate gradient of walks using fixed point math
        xflag = fabs(a) > fabs(b)
        if xflag:
            if a > 0:
                dx0 = 1
            else:
                dx0 = -1
            dy0 = round(b * (1 << shift) / fabs(a))
            y0 = (y0 << shift) + (1 << (shift - 1))
        else:
            if b > 0:
                dy0 = 1
            else:
                dy0 = -1
            dx0 = round(a * (1 << shift) / fabs(b))
            x0 = (x0 << shift) + (1 << (shift - 1))

        # pass 1: walk the line, merging lines less than specified gap
        # length
        for k in range(2):
            gap = 0
            px = x0
            py = y0
            dx = dx0
            dy = dy0
            if k > 0:
                dx = -dx
                dy = -dy
            while 1:
                if xflag:
                    x1 = px
                    y1 = py >> shift
                else:
                    x1 = px >> shift
                    y1 = py
                # check when line exits image boundary
                if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
                    break
                gap += 1
                # if non-zero point found, continue the line
                if mask[y1, x1]:
                    gap = 0
                    line_end[2*k] = x1
                    line_end[2*k + 1] = y1
                # if gap to this point was too large, end the line
                elif gap > line_gap:
                    break
                px += dx
                py += dy

        # confirm line length is sufficient
        good_line = (abs(line_end[3] - line_end[1]) >= line_length or
                     abs(line_end[2] - line_end[0]) >= line_length)

        # pass 2: walk the line again and reset accumulator and mask
        for k in range(2):
            px = x0
            py = y0
            dx = dx0
            dy = dy0
            if k > 0:
                dx = -dx
                dy = -dy
            while 1:
                if xflag:
                    x1 = px
                    y1 = py >> shift
                else:
                    x1 = px >> shift
                    y1 = py
                # if non-zero point found, continue the line
                if mask[y1, x1]:
                    if good_line:
                        accum_idx = round(
                            (ctheta[j] * x1 + stheta[j] * y1)) + offset
                        # accum[accum_idx, max_theta] -= 1
                        accum[accum_idx, max_theta] -= img[y1, x1];

                        mask[y1, x1] = 0
                # exit when the point is the line end
                if x1 == line_end[2*k] and y1 == line_end[2*k + 1]:
                    break
                px += dx
                py += dy

        # add line to the result
        if good_line:
            lines[nlines, 0, 0] = line_end[0]
            lines[nlines, 0, 1] = line_end[1]
            lines[nlines, 1, 0] = line_end[2]
            lines[nlines, 1, 1] = line_end[3]
            nlines += 1
            if nlines >= lines_max:
                break

    PyMem_Free(line_end)
    return [((line[0, 0], line[0, 1]), (line[1, 0], line[1, 1]))
            for line in lines[:nlines]]
