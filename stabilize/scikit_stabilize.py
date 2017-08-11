#!/usr/bin/env python
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
from os import path, mkdir, listdir
from joblib import Parallel, delayed
from skimage.feature import register_translation
from skimage import transform, io
from skimage.color import rgb2gray
from skimage.filters import sobel


def get_image_list(pattern='alpes_%d.jpg', start=0):
    """
    Reads a list of images given a pattern.
    :param pattern: A pattern containing one integer, e.g. "dir/image_%d.jpg".
    :return:        A list of existing image file names.
    """
    image_list = []
    k = start
    while path.exists(pattern % k):
        image_list.append(pattern % k)
        k += 1
    return image_list


def get_files_from_dir(dir_name):
    """
    Returns the list of files in a directory
    :param dir_name: The directory to be listed
    :return:         A list of paths to files in the directory
    """
    return [path.join(dir_name, f) for f in listdir(dir_name) if path.isfile(path.join(dir_name, f))]


def prepare(img):
    """
    Pre-process the image before translation detection, here we transform to black and white and use edge-detection.
    :param img: An image (as numpy array)
    :return: The preprocessed image (as numpy array)
    """
    return sobel(rgb2gray(img))


def correct(img_src, shift, destfile, crop=None):
    """
    Apply the correction, crop and saves to a file.
    :param img_src:  The source image
    :param shift:    The shift to be corrected
    :param destfile: The name of the destination file
    :param crop:     The crop to be applied
    :return:         The destination filename
    """
    (y0, y1, x0, x1) = crop
    (shift_y, shift_x) = shift
    tf_shift = transform.SimilarityTransform(translation=[shift_x, shift_y])
    img = transform.warp(img_src, tf_shift)
    res = img[-y0:-y1, -x0:-x1, :]
    io.imsave(destfile, res)
    return destfile


def find_shift(ref, img):
    """
    Find a translation between two images
    :param ref: The reference image
    :param img: The image
    :return:    The shift
    """
    im0 = prepare(ref)
    im1 = prepare(img)
    shift, error, diffphase = register_translation(im0, im1, 100)

    return shift


def batch_align(image_list, dest_dir="output"):
    """
    Correct the sharking on the series of images
    :param image_list: The input series of images
    :param dest_dir:   The destination directory
    """
    if not path.exists(dest_dir):
        mkdir(dest_dir)
    if path.isdir(dest_dir):
        print "Aligning %d images, output in %s, this may take a while" % (len(im_list), dest_dir)

        ref_img = io.imread(image_list[0])
        r = Parallel(n_jobs=4, backend="threading", verbose=25)(
            delayed(find_shift)(io.imread(img), ref_img) for img in image_list[1:])
        y_shift = map(lambda x: x[0], r)
        x_shift = map(lambda x: x[1], r)

        print min(y_shift), max(y_shift), min(x_shift), max(x_shift)
        crop = [int(min(y_shift)) - 1, int(max(y_shift)) + 1, int(min(x_shift)) - 1, int(max(x_shift)) + 1]

        correct(ref_img, (0, 0), "%s/%s" % (dest_dir, path.basename(image_list[0])), crop)
        Parallel(n_jobs=4, backend="threading", verbose=25)(
            delayed(correct)(io.imread(img), r[k], "%s/%s" % (dest_dir, path.basename(image_list[k])), crop)
            for k, img in enumerate(image_list[1:]))
    else:
        print "Output dir does not exists or is not a directory : %s" % dest_dir


def usage():
    print """Usage: ./deshake.py INPUT_DIR [OUTPUT_DIR]
Notes:
    - The INPUT_DIR must contain only image files.
    - The file names are kept in the OUTPUT_DIR
    - "output" is used by default, when no other dir is given for the output."""
    exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 1 or len(sys.argv) > 3:
        usage()
    else:
        input_dir = sys.argv[1]
        im_list = get_files_from_dir(input_dir)
        if len(sys.argv) > 2:
            output = sys.argv[2]
            batch_align(im_list, output)
        else:
            batch_align(im_list)