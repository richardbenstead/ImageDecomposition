# probabilistic hough transform image decomposition
import numpy as np

from skimage.transform import hough_line_peaks
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.draw import line_aa
from skimage import data
from skimage import io

import matplotlib.pyplot as plt
from matplotlib import cm

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
from hough.hough import _hough_line
from hough.hough import _probabilistic_hough_line

# Line finding using the Probabilistic Hough Transform
# image = data.camera()
# image = data.horse()
# image = exposure.equalize_hist(image)
image = io.imread('stiller.jpg', as_gray=True)
#image = io.imread('eye.jpg', as_gray=True)
# image = io.imread('gorilla.jpg', as_gray=True)


if False:
    image = np.ones((200, 200))*255
    idx = np.arange(25, 175)
    image[idx[::-1], idx] = 0
    image[10+idx[::-1], idx] = 0
    image[np.arange(200), np.arange(200)] = 0
    image[np.arange(100)+50, np.arange(100)+50] = 0
    image[15, idx-20] = 0  # horizonal line, y,x

if image.dtype==np.dtype(bool) or np.max(image) <= 1:
    image = (image.astype(float)*255).astype(int)

raw_image = image.copy()

line_image = image * 0
line_image[:,:]=255
blur_line_image = line_image.copy()

init_angle = 0
theta=np.linspace(init_angle, init_angle + np.pi, 5, endpoint=False)

# need to keep original white (orig_image) - should penalize
# don't need to cover what is already covered (current_image)

length = np.sqrt(image.shape[1] ** 2 + image.shape[0] ** 2) * 1
adjust_image = image * 0

line_length = np.sqrt(image.shape[1] ** 2 + image.shape[0] ** 2) / 2
for j in range(20):
    for i in range(20):
        theta += np.pi/10
        print("Getting transform")
        lines = _probabilistic_hough_line(image, threshold=1, line_length=int(line_length), line_gap=4, theta=theta)

        print("Plotting lines")
        np.random.shuffle(lines)

        added=0
        # blur_line_image = gaussian(line_image, sigma=20, preserve_range=True)
        line_len = [-((e[1] + s[1])**2 + (e[0] + s[0])**2) for (s,e) in lines]
        ind = np.argsort(line_len)[:1]
        for i in range(min(20, len(lines))):
            line = lines[i]
            p0, p1 = line

            # get anti-aliased line points
            rr, cc, vv = line_aa(p0[0], p0[1], p1[0],p1[1])
            sum_dupe=0
            for r,c,v in zip(rr,cc,vv):
                sum_dupe += blur_line_image[c, r] < 100

            if sum_dupe/len(rr) < 0.1:
                added += 1
                for r,c,v in zip(rr,cc,vv):
                    # dark target pixels that were hit should be set to 128 - so don't need to hit again
                    # light pixels that were hit, should stay light (to avoid)
                    adjust_image[c, r] = 255 # max(200, image[c, r])
                    line_image[c, r] = 0 # max(0, line_image[c, r] - v*200)
                blur_line_image = gaussian(line_image, sigma=5, preserve_range=True)

        print(f'added {added} lines out of {len(lines)}')

        blur_adjust = gaussian(adjust_image, sigma=10, preserve_range=True) * 2
        blur_adjust = 255 * blur_adjust / np.max(blur_adjust)

        image = raw_image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j] = max(blur_adjust[i,j], image[i,j])

        line_length *= 0.95
        print(f'Line length {line_length}')

    # plot images
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(blur_adjust, cmap=cm.gray)
    ax[1].set_title('adjust image')
    ax[1].set_axis_off()

    ax[2].imshow(line_image, cmap=cm.gray)
    ax[2].set_axis_off()

    plt.show()
