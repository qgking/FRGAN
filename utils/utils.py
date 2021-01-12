import sys
import os
from os.path import join
import nibabel as nib
import scipy.misc
from scipy import ndimage as nd
from glob import glob
from skimage.measure import label as mlabel
import numpy as np
from os import fsync, makedirs
from PIL import Image
from scipy.ndimage import zoom as cpzoom
from scipy.ndimage import rotate as cprotate
import scipy.ndimage.morphology as morph
import scipy.ndimage.filters as filters
import math
from medpy import metric
from scipy import ndimage
from hausdorff import hausdorff_distance
import zipfile
from torchvision.utils import make_grid
from skimage.transform import resize
from scipy import linalg
from datetime import datetime

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8
TMP_DIR = "./tmp"
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ====================================================
# ======================volume preprocessing method===
# ====================================================
def to_scale(img, slice_shape, shape=None):
    if shape is None:
        shape = slice_shape

    height, width = shape
    if img.dtype == SEG_DTYPE:
        return scipy.misc.imresize(img, (height, width), interp="nearest").astype(SEG_DTYPE)
    elif img.dtype == IMG_DTYPE:
        factor = 256.0 / np.max(img)
        return (scipy.misc.imresize(img, (height, width), interp="nearest") / factor).astype(IMG_DTYPE)
    else:
        raise TypeError(
            'Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')


def resample(img, seg, scan, new_voxel_dim=[1, 1, 1]):
    # Get voxel size
    voxel_dim = np.array(scan.header.structarr["pixdim"][1:4], dtype=np.float32)
    # Resample to optimal [1,1,1] voxel size
    resize_factor = voxel_dim / new_voxel_dim
    scan_shape = np.array(scan.header.get_data_shape())
    new_scan_shape = scan_shape * resize_factor
    rounded_new_scan_shape = np.round(new_scan_shape)
    print('unique before resample: ', np.unique(seg))
    print("new shape ", rounded_new_scan_shape)
    rounded_resize_factor = rounded_new_scan_shape / scan_shape  # Change resizing due to round off error
    new_voxel_dim = voxel_dim / rounded_resize_factor

    img = resize(img, rounded_new_scan_shape, order=0, mode='edge',
                 cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    seg = resize(seg, rounded_new_scan_shape, order=3, mode='constant',
                 cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    # img = nd.interpolation.zoom(img, rounded_resize_factor, mode='nearest')
    # seg = nd.interpolation.zoom(seg, rounded_resize_factor, mode='nearest')
    print('unique after resample: ', np.unique(seg))
    seg = np.round(seg)
    print('unique after round: ', np.unique(seg))
    return img, seg, new_voxel_dim


def resample_numpy(img, seg, old_dim, new_voxel_dim=[1, 1, 1]):
    # Get voxel size
    voxel_dim = np.array(old_dim)
    # Resample to optimal [1,1,1] voxel size
    resize_factor = voxel_dim / new_voxel_dim
    scan_shape = np.array(img.shape)
    new_scan_shape = scan_shape * resize_factor
    rounded_new_scan_shape = np.round(new_scan_shape)
    print("new shape ", rounded_new_scan_shape)
    rounded_resize_factor = rounded_new_scan_shape / scan_shape  # Change resizing due to round off error
    new_voxel_dim = voxel_dim / rounded_resize_factor

    img = resize(img, rounded_new_scan_shape, order=0, mode='edge',
                 cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    seg = resize(seg, rounded_new_scan_shape, order=3, mode='constant',
                 cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    # img = nd.interpolation.zoom(img, rounded_resize_factor, mode='nearest')
    # seg = nd.interpolation.zoom(seg, rounded_resize_factor, mode='nearest')
    print('unique after resample: ', np.unique(seg))
    seg = np.round(seg)
    print('unique after round: ', np.unique(seg))
    return img, seg, new_voxel_dim


def norm_hounsfield_dyn(arr, c_min=0.1, c_max=0.3):
    """ Converts from hounsfield units to float64 image with range 0.0 to 1.0 """
    # calc min and max
    min, max = np.amin(arr), np.amax(arr)
    if min <= 0:
        arr = np.clip(arr, min * c_min, max * c_max)
        # right shift to zero
        arr = np.abs(min * c_min) + arr
    else:
        arr = np.clip(arr, min, max * c_max)
        # left shift to zero
        arr = arr - min
    # normalization
    norm_fac = np.amax(arr)
    if norm_fac != 0:
        norm = np.divide(
            np.multiply(arr, 255),
            np.amax(arr))
    else:  # don't divide through 0
        norm = np.multiply(arr, 255)

    norm = np.clip(np.multiply(norm, 0.00390625), 0, 1)
    return norm


def histeq_processor(img):
    """Histogram equalization"""
    nbr_bins = 256
    # get image histogram
    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    original_shape = img.shape
    img = np.interp(img.flatten(), bins[:-1], cdf)
    img = img / 256.0
    return img.reshape(original_shape)


# Get majority label in image
def largest_label_volume(img, bg=0):
    vals, counts = np.unique(img, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def label_connected_component(pred):
    seg = mlabel(pred, neighbors=8, background=0)
    return seg


# brain utils
def generate_patch_locations(patches, patch_size, im_size):
    nx = round((patches * 8 * im_size[0] * im_size[0] / im_size[1] / im_size[2]) ** (1.0 / 3))
    ny = round(nx * im_size[1] / im_size[0])
    nz = round(nx * im_size[2] / im_size[0])
    x = np.rint(np.linspace(patch_size, im_size[0] - patch_size, num=nx))
    y = np.rint(np.linspace(patch_size, im_size[1] - patch_size, num=ny))
    z = np.rint(np.linspace(patch_size, im_size[2] - patch_size, num=nz))
    return x, y, z


def perturb_patch_locations(patch_locations, radius):
    x, y, z = patch_locations
    x = np.rint(x + np.random.uniform(-radius, radius, len(x)))
    y = np.rint(y + np.random.uniform(-radius, radius, len(y)))
    z = np.rint(z + np.random.uniform(-radius, radius, len(z)))
    return x, y, z


def generate_patch_probs(path, patch_locations, patch_size, im_size, tagTumor=0):
    x, y, z = patch_locations
    seg = nib.load(glob(os.path.join(path, '*_seg.nii.gz'))[0]).get_data().astype(np.float32)
    p = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                patch = seg[int(x[i] - patch_size / 2): int(x[i] + patch_size / 2),
                        int(y[j] - patch_size / 2): int(y[j] + patch_size / 2),
                        int(z[k] - patch_size / 2): int(z[k] + patch_size / 2)]
                patch = (patch > tagTumor).astype(np.float32)
                percent = np.sum(patch) / (patch_size * patch_size * patch_size)
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype=np.float32)
    p[p == 0] = np.amin(p[np.nonzero(p)])
    p = p / np.sum(p)
    return p


def normalize_roi(im_input):
    x_start = im_input.shape[0] // 4
    x_range = im_input.shape[0] // 2
    y_start = im_input.shape[1] // 4
    y_range = im_input.shape[1] // 2
    z_start = im_input.shape[2] // 4
    z_range = im_input.shape[2] // 2
    roi = im_input[x_start: x_start + x_range, y_start: y_start + y_range, z_start: z_start + z_range]
    roi = (im_input - np.mean(roi)) / np.std(roi)
    # rescale to 0  1
    im_output = (roi - np.min(roi)) / (np.max(roi) - np.min(roi))
    return im_output


def read_image(path, is_training=True):
    t1 = nib.load(glob(os.path.join(path, '*_t1.nii.gz'))[0]).get_data().astype(np.float32)
    t1ce = nib.load(glob(os.path.join(path, '*_t1ce.nii.gz'))[0]).get_data().astype(np.float32)
    t2 = nib.load(glob(os.path.join(path, '*_t2.nii.gz'))[0]).get_data().astype(np.float32)
    flair = nib.load(glob(os.path.join(path, '*_flair.nii.gz'))[0]).get_data().astype(np.float32)
    assert t1.shape == t1ce.shape == t2.shape == flair.shape
    if is_training:
        seg = nib.load(glob(os.path.join(path, '*_seg.nii.gz'))[0]).get_data().astype(np.float32)
        assert t1.shape == seg.shape
        nchannel = 5
    else:
        nchannel = 4

    image = np.empty((t1.shape[0], t1.shape[1], t1.shape[2], nchannel), dtype=np.float32)

    # image[..., 0] = remove_low_high(t1)
    # image[..., 1] = remove_low_high(t1ce)
    # image[..., 2] = remove_low_high(t2)
    # image[..., 3] = remove_low_high(flair)
    image[..., 0] = normalize_roi(t1)
    image[..., 1] = normalize_roi(t1ce)
    image[..., 2] = normalize_roi(t2)
    image[..., 3] = normalize_roi(flair)

    if is_training:
        image[..., 4] = np.clip(seg, 0, 1)

    return image


def generate_test_locations(patch_size, stride, im_size):
    stride_size_x = patch_size[0] / stride
    stride_size_y = patch_size[1] / stride
    stride_size_z = patch_size[2] / stride
    pad_x = (
        int(patch_size[0] / 2),
        int(np.ceil(im_size[0] / stride_size_x) * stride_size_x - im_size[0] + patch_size[0] / 2))
    pad_y = (
        int(patch_size[1] / 2),
        int(np.ceil(im_size[1] / stride_size_y) * stride_size_y - im_size[1] + patch_size[1] / 2))
    pad_z = (
        int(patch_size[2] / 2),
        int(np.ceil(im_size[2] / stride_size_z) * stride_size_z - im_size[2] + patch_size[2] / 2))
    x = np.arange(patch_size[0] / 2, im_size[0] + pad_x[0] + pad_x[1] - patch_size[0] / 2 + 1, stride_size_x)
    y = np.arange(patch_size[1] / 2, im_size[1] + pad_y[0] + pad_y[1] - patch_size[1] / 2 + 1, stride_size_y)
    z = np.arange(patch_size[2] / 2, im_size[2] + pad_z[0] + pad_z[1] - patch_size[2] / 2 + 1, stride_size_z)
    return (x, y, z), (pad_x, pad_y, pad_z)


def min_max_voi(mask, superior=10, inferior=10):
    sp = mask.shape
    tp = np.transpose(np.nonzero(mask))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    minz = 0 if minz - superior < 0 else minz - superior
    maxz = sp[-1] if maxz + inferior > sp[-1] else maxz + inferior + 1
    miny = 0 if miny - superior < 0 else miny - superior
    maxy = sp[1] if maxy + inferior > sp[1] else maxy + inferior + 1
    minx = 0 if minx - superior < 0 else minx - superior
    maxx = sp[0] if maxx + inferior > sp[0] else maxx + inferior + 1
    return minx, maxx, miny, maxy, minz, maxz


# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


def normalize_(img):
    # imgs_normalized = (img - np.mean(img)) / np.std(img)
    # imgs_normalized =img
    imgs_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    return imgs_normalized


def overlay(volume_ims, segmentation_ims, segmentation, alpha=0.3, k_color=[255, 0, 0], t_color=[0, 0, 255]):
    im_volume = 255 * volume_ims
    volume_ims = np.stack((im_volume, im_volume, im_volume), axis=-1)
    segmentation_ims = class_to_color(segmentation_ims, k_color=k_color, t_color=t_color)
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha * segmentation_ims + (1 - alpha) * volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation, 1)] = k_color
    # seg_color[np.equal(segmentation, 2)] = t_color
    return seg_color


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def cal_ssim(im1, im2):
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def cal_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# from models.inception import InceptionV3
# def evaluate_fid_score(root_gt_dir, root_result_dir, result_postfix):
#     result_dirs = get_everything_under(root_result_dir, only_dirs=True)
#     gt_dirs = get_everything_under(root_gt_dir, only_dirs=True)
#
#     output_i3d_activations = []
#     real_i3d_activations = []
#
#     with torch.no_grad():
#         for i, (result_dir, gt_dir) in enumerate(zip(result_dirs, gt_dirs)):
#             result_dir = os.path.join(result_dir, result_postfix)
#             result_frame_reader = FrameReader(result_dir).files
#             gt_frame_reader = FrameReader(gt_dir).files
#
#             # Unsqueeze batch dimension
#             outputs = to_tensors(result_frame_reader).unsqueeze(0).to(torch.device('cuda:0'))
#             targets = to_tensors(gt_frame_reader).unsqueeze(0).to(torch.device('cuda:0'))
#             # get i3d activation
#             output_i3d_activations.append(get_i3d_activations(outputs).cpu().numpy())
#             real_i3d_activations.append(get_i3d_activations(targets).cpu().numpy())
#         # concat and evaluate fid score
#         output_i3d_activations = np.concatenate(output_i3d_activations, axis=0)
#         real_i3d_activations = np.concatenate(real_i3d_activations, axis=0)
#         fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
#     logger.info(f"Video num: {len(result_dirs)}")
#     logger.info(f"FID score: {fid_score}")
#     return fid_score

def cal_fid_score(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, rowvar=False)
    s2 = np.cov(fake_activations, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)


# code from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)


def gen_patches(image, seg, seg_unique, patch_per_modality, padding, new_shape, rootdir, vol, thres=30, filter=2):
    rotate_angle = [0, 90, 180, 270]
    # rotate_axis = [(1, 0), (1, 2), (2, 0)]
    rotate_axis = [(1, 0)]

    for unique in seg_unique:
        seg_uni = seg.copy()
        # image_uni = image.copy()
        total_pixels = np.sum(seg_uni == unique)
        if total_pixels < thres:
            print("total pixels less skip")
            continue
        print("total pixels:" + str(total_pixels))
        # image_uni[np.where(seg_uni == unique)] = 1
        seg_uni = np.where(seg_uni != unique, 0, 1)
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=0, inferior=0)
        print((maxx - minx, maxy - miny, maxz - minz))

        # image[minx:maxx, miny:maxy, minz: maxz, 0:4] = 1
        # generate cube mask
        # imgtmp = image[minx:maxx, miny:maxy, minz: maxz, -2]
        # seg_uni = seg_uni[minx:maxx, miny:maxy, minz: maxz]
        # tsp = seg_uni.shape
        # x = np.linspace(0, tsp[0], tsp[0], endpoint=False)
        # y = np.linspace(0, tsp[1], tsp[1], endpoint=False)
        # z = np.linspace(0, tsp[2], tsp[2], endpoint=False)
        # # Manipulate x,y,z here to obtain the dimensions you are looking for
        #
        # center = np.array([tsp[0] // 2, tsp[1] // 2, tsp[2] // 2])
        #
        # # Generate grid of points
        # X, Y, Z = np.meshgrid(x, y, z)
        # data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        #
        # distance = sp.distance.cdist(data, center.reshape(1, -1)).ravel()
        # points_in_sphere = data[distance < np.max(center)]

        # from copy import deepcopy
        #
        # ''' size : size of original 3D numpy matrix A.
        #     radius : radius of circle inside A which will be filled with ones.
        # '''
        # size, radius = 5, 2
        #
        # ''' A : numpy.ndarray of shape size*size*size. '''
        # A = np.zeros((size, size, size))
        #
        # ''' AA : copy of A (you don't want the original copy of A to be overwritten.) '''
        # AA = deepcopy(A)
        #
        # ''' (x0, y0, z0) : coordinates of center of circle inside A. '''
        # x0, y0, z0 = int(np.floor(A.shape[0] / 2)), \
        #              int(np.floor(A.shape[1] / 2)), int(np.floor(A.shape[2] / 2))
        #
        # for x in range(x0 - radius, x0 + radius + 1):
        #     for y in range(y0 - radius, y0 + radius + 1):
        #         for z in range(z0 - radius, z0 + radius + 1):
        #             ''' deb: measures how far a coordinate in A is far from the center.
        #                     deb>=0: inside the sphere.
        #                     deb<0: outside the sphere.'''
        #             deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
        #             if (deb) >= 0: AA[x, y, z] = 1

        for num in range(patch_per_modality):
            mminx = np.random.randint(minx - padding, minx)
            mmaxx = np.random.randint(maxx, maxx + padding)

            mminy = np.random.randint(miny - padding, miny)
            mmaxy = np.random.randint(maxy, maxy + padding)

            mminz = np.random.randint(minz - padding // 2, minz)
            mmaxz = np.random.randint(maxz, maxz + padding // 2)

            # mminx = minx - padding
            mminx = mminx if mminx > 0 else 0
            # mmaxx = maxx + padding
            # mminy = miny - padding
            mminy = mminy if mminy > 0 else 0
            # mmaxy = maxy + padding
            # mminz = minz - padding // 4
            mminz = mminz if mminz > 0 else 0
            # mmaxz = maxz + padding // 4
            mmaxz = mmaxz if mmaxz < image.shape[2] else image.shape[2] - 1

            # for fit the network, two downsample and concat
            # mmaxz = mmaxz - (mmaxz - mminz) % 4
            # mmaxy = mmaxy - (mmaxy - mminy) % 4
            # mmaxx = mmaxx - (mmaxx - mminx) % 4
            print((mminx, mmaxx, mminy, mmaxy, mminz, mmaxz))

            ori_npy = image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
            seg_npy = seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
            # boudray sigma 1.0      tumor sigma 5.0
            # boundary_npy = ((
            #                     filters.gaussian_filter(
            #                         255. * np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy),
            #                         1.0)) / 255.).reshape(seg_npy.shape)

            # boundary_npy = np.where(np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy) > 0, 1, 0)
            # tumor_npy = ((
            #                  filters.gaussian_filter(255. * np.multiply(morph.binary_erosion(seg_npy), seg_npy),
            #                                          3.0)) / 255).reshape(seg_npy.shape)
            # tumor_npy = np.multiply(morph.binary_erosion(seg_npy), seg_npy).reshape(seg_npy.shape)
            resize_factor = np.array(new_shape, dtype=np.float32) / [ori_npy.shape[0], ori_npy.shape[1],
                                                                     ori_npy.shape[2]]
            ori_npy = cpzoom(ori_npy, resize_factor, order=1)
            seg_npy = cpzoom(seg_npy, resize_factor, order=1)
            if np.random.random() > 0.5:
                rotate_a = np.random.randint(len(rotate_angle))
                rotate_b = np.random.randint(len(rotate_axis))
                angle = rotate_angle[rotate_a]
                raxis = rotate_axis[rotate_b]
                ori_npy = cprotate(ori_npy, angle, raxis, order=1)
                seg_npy = cprotate(seg_npy, angle, raxis, order=1)

            # ori_npy = nd.interpolation.zoom(ori_npy, resize_factor, mode='wrap')
            # seg_npy = nd.interpolation.zoom(seg_npy, resize_factor, mode='wrap')
            # create boundary
            seg_npy = np.where(seg_npy > 0, 1, 0)
            boundary_npy = ((
                                filters.gaussian_filter(
                                    255. * np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy),
                                    1.0)) / 255).reshape(seg_npy.shape)
            boundary_npy = np.where(boundary_npy > np.max(boundary_npy) / filter, 1, 0)

            # create tumor region
            tumor_npy = seg_npy.copy()

            # create mask,normalization
            # ori_npy = normalize_(ori_npy)
            msk_npy = ori_npy.copy()
            msk_npy[seg_npy > 0] = 1

            save_file = np.stack([ori_npy, msk_npy, tumor_npy, boundary_npy, seg_npy], -1)
            # for sss in range(save_file.shape[-2]):
            #     visualize(np.expand_dims(save_file[..., 0][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Araw"))
            #     visualize(np.expand_dims(save_file[..., 1][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Berase"))
            #     visualize(np.expand_dims(save_file[..., 2][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss)+ "Ctumor"))
            #     visualize(np.expand_dims(save_file[..., 3][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Dboundary"))
            #     visualize(np.expand_dims(save_file[..., 4][:, :, sss], -1),
            #               join(rootdir, 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
            #                   unique) + '_' + '{0:0>2}'.format(
            #                   num) + str(sss) + "Eseg"))
            np.save(
                rootdir + 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
                    unique) + '_' + '{0:0>2}'.format(
                    num) + '.npy', save_file)


def gen_patches_2d(image, seg, seg_unique, patch_per_modality, padding, new_shape, rootdir, vol, thres=30, filter=2):
    rotate_angle = [0, 90, 180, 270]
    rotate_axis = [(1, 0), (1, 2), (2, 0)]
    for unique in seg_unique:
        seg_uni = seg.copy()
        # image_uni = image.copy()
        total_pixels = np.sum(seg_uni == unique)
        if total_pixels < thres:
            print("total pixels less skip")
            continue
        print("total pixels:" + str(total_pixels))
        # image_uni[np.where(seg_uni == unique)] = 1
        seg_uni = np.where(seg_uni != unique, 0, 1)
        minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=0, inferior=0)
        print((maxx - minx, maxy - miny, maxz - minz))
        # TODO delete
        # seg_uni[minx:maxx, miny:maxy, minz:maxz] = 1

        # tmp = np.where(seg_uni > 0, 1, 0)
        # for sss in range(image_uni.shape[-1]):
        #     visualize(np.expand_dims(image[..., sss], -1),
        #               join(TMP_DIR, str(sss) + "Araw"))
        #     visualize(np.expand_dims(image_uni[..., sss], -1),
        #               join(TMP_DIR, str(sss) + "Berase"))
        #     visualize(np.expand_dims(tmp[..., sss], -1),
        #               join(TMP_DIR, str(sss) + "Cseg"))

        # image[minx:maxx, miny:maxy, minz: maxz, 0:4] = 1
        # generate cube mask
        # imgtmp = image[minx:maxx, miny:maxy, minz: maxz, -2]
        # seg_uni = seg_uni[minx:maxx, miny:maxy, minz: maxz]
        # tsp = seg_uni.shape
        # x = np.linspace(0, tsp[0], tsp[0], endpoint=False)
        # y = np.linspace(0, tsp[1], tsp[1], endpoint=False)
        # z = np.linspace(0, tsp[2], tsp[2], endpoint=False)
        # # Manipulate x,y,z here to obtain the dimensions you are looking for
        #
        # center = np.array([tsp[0] // 2, tsp[1] // 2, tsp[2] // 2])
        #
        # # Generate grid of points
        # X, Y, Z = np.meshgrid(x, y, z)
        # data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        #
        # distance = sp.distance.cdist(data, center.reshape(1, -1)).ravel()
        # points_in_sphere = data[distance < np.max(center)]

        # from copy import deepcopy
        #
        # ''' size : size of original 3D numpy matrix A.
        #     radius : radius of circle inside A which will be filled with ones.
        # '''
        # size, radius = 5, 2
        #
        # ''' A : numpy.ndarray of shape size*size*size. '''
        # A = np.zeros((size, size, size))
        #
        # ''' AA : copy of A (you don't want the original copy of A to be overwritten.) '''
        # AA = deepcopy(A)
        #
        # ''' (x0, y0, z0) : coordinates of center of circle inside A. '''
        # x0, y0, z0 = int(np.floor(A.shape[0] / 2)), \
        #              int(np.floor(A.shape[1] / 2)), int(np.floor(A.shape[2] / 2))
        #
        # for x in range(x0 - radius, x0 + radius + 1):
        #     for y in range(y0 - radius, y0 + radius + 1):
        #         for z in range(z0 - radius, z0 + radius + 1):
        #             ''' deb: measures how far a coordinate in A is far from the center.
        #                     deb>=0: inside the sphere.
        #                     deb<0: outside the sphere.'''
        #             deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
        #             if (deb) >= 0: AA[x, y, z] = 1

        for num in range(patch_per_modality):
            # mminx = np.random.randint(minx - padding, minx)
            # mmaxx = np.random.randint(maxx, maxx + padding)
            # mminy = np.random.randint(miny - padding, miny)
            # mmaxy = np.random.randint(maxy, maxy + padding)
            # mminz = np.random.randint(minz - padding, minz)
            mminx = np.random.randint(minx - padding, minx)
            mmaxx = np.random.randint(maxx, maxx + padding)
            mminy = np.random.randint(miny - padding, miny)
            mmaxy = np.random.randint(maxy, maxy + padding)

            mminx = mminx if mminx > 0 else 0
            mminy = mminy if mminy > 0 else 0
            mminz = minz
            mmaxz = maxz

            # mminx = minx - padding
            # mminx = mminx if mminx > 0 else 0
            # mmaxx = maxx + padding
            # mminy = miny - padding
            # mminy = mminy if mminy > 0 else 0
            # mmaxy = maxy + padding
            # mminz = minz
            # mmaxz = maxz
            print((mminx, mmaxx, mminy, mmaxy, mminz, mmaxz))
            ori_npy = image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
            seg_npy = seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]

            boundary_npy = ((
                                filters.gaussian_filter(
                                    255. * np.multiply(np.invert(morph.binary_erosion(seg_npy)), seg_npy),
                                    1.0)) / 255).reshape(seg_npy.shape)
            boundary_npy = np.where(boundary_npy > np.max(boundary_npy) / filter, 1, 0)
            tumor_npy = seg_npy.copy()
            new_shape[-1] = ori_npy.shape[2]
            resize_factor = np.array(new_shape, dtype=np.float32) / [ori_npy.shape[0], ori_npy.shape[1],
                                                                     ori_npy.shape[2]]
            ori_npy = cpzoom(ori_npy, resize_factor, order=1)
            seg_npy = cpzoom(seg_npy, resize_factor, order=1)
            boundary_npy = cpzoom(boundary_npy, resize_factor, order=1)
            tumor_npy = cpzoom(tumor_npy, resize_factor, order=1)

            # ori_npy = nd.interpolation.zoom(ori_npy, resize_factor, mode='wrap')
            # msk_npy = nd.interpolation.zoom(msk_npy, resize_factor, mode='wrap')
            # seg_npy = nd.interpolation.zoom(seg_npy, resize_factor, mode='wrap')
            msk_npy = ori_npy.copy()
            msk_npy[seg_npy > 0] = 1
            save_file = np.stack([ori_npy, msk_npy, tumor_npy, boundary_npy, seg_npy], -1)
            for sss in range(save_file.shape[-2]):
                total_pixels = np.sum(save_file[..., 2][:, :, sss] > 0)
                if total_pixels < 15:
                    print("total pixels less skip")
                    continue
                save_file_2d = np.stack(
                    [save_file[..., 0][:, :, sss], save_file[..., 1][:, :, sss], save_file[..., 2][:, :, sss],
                     save_file[..., 3][:, :, sss], save_file[..., 4][:, :, sss]], -1)
                np.save(
                    rootdir + 'img_' + '{0:0>3}'.format(vol) + '_' + '{0:0>1}'.format(
                        unique) + '_' + '{0:0>2}'.format(
                        sss) + '.npy', save_file_2d)
                # visualize(np.expand_dims(save_file[..., 0][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Araw"))
                # visualize(np.expand_dims(save_file[..., 1][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Berase"))
                # visualize(np.expand_dims(save_file[..., 2][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Ctumor"))
                # visualize(np.expand_dims(save_file[..., 3][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Dboundary"))
                # visualize(np.expand_dims(save_file[..., 4][:, :, sss], -1),
                #           join(TMP_DIR, str(sss) + "Eseg"))


def compute_segmentation_scores(prediction_mask, reference_mask):
    """
    Calculates metrics scores from numpy arrays and returns an dict.

    Assumes that each object in the input mask has an integer label that
    defines object correspondence between prediction_mask and
    reference_mask.

    :param prediction_mask: numpy.array, int
    :param reference_mask: numpy.array, int
    :param voxel_spacing: list with x,y and z spacing
    :return: dict with dice, jaccard, voe, rvd, assd, rmsd, and msd
    """

    scores = {'dice': [],
              'jaccard': [],
              'voe': [],
              'rvd': []}

    for i, obj_id in enumerate(np.unique(prediction_mask)):
        if obj_id == 0:
            continue  # 0 is background, not an object; skip

        # Limit processing to the bounding box containing both the prediction
        # and reference objects.
        target_mask = (reference_mask == obj_id) + (prediction_mask == obj_id)
        bounding_box = ndimage.find_objects(target_mask)[0]
        p = (prediction_mask == obj_id)[bounding_box]
        r = (reference_mask == obj_id)[bounding_box]
        if np.any(p) and np.any(r):
            dice = metric.dc(p, r)
            jaccard = dice / (2. - dice)
            scores['dice'].append(dice)
            scores['jaccard'].append(jaccard)
            scores['voe'].append(1. - jaccard)
            scores['rvd'].append(metric.ravd(r, p))
    return scores


def evaluate_metric(data_output, data_target):
    scores = {
        'mse': [],
        'rmse': [],
        'hd': []}
    for j in range(data_output.shape[0]):
        X = data_output[j, :, :]
        Y = data_target[j, :, :]
        mse = (((np.sum((X - Y) ** 2)) / (data_output.shape[1] * data_output.shape[2])))
        rmse = np.sqrt(np.sum((X - Y) ** 2) / (data_output.shape[1] * data_output.shape[2]))
        scores['mse'].append(mse)
        scores['rmse'].append(rmse)
        tmp_hausdorff = hausdorff_distance(X, Y)
        scores['hd'].append(tmp_hausdorff)
    scores_final = {
        'mse': [],
        'rmse': [],
        'hd': []}
    scores_final['mse'].append(np.mean(scores['mse']))
    scores_final['rmse'].append(np.mean(scores['rmse']))
    scores_final['hd'].append(np.mean(scores['hd']))
    return scores_final


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def visual_batch(batch, dir, name, channel=1, nrow=8):
    batch_len = len(batch.size())
    if batch_len == 3:
        image_save = batch.detach().contiguous()
        image_save = image_save.unsqueeze(1)
        grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
    if batch_len == 4:
        if channel == 3:
            image_save = batch.detach().contiguous()
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
        else:
            image_save = batch.detach().contiguous()
            image_save = image_save.view(
                (image_save.size(0) * image_save.size(1), image_save.size(2), image_save.size(3)))
            image_save = image_save.unsqueeze(1)
            grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
            save_img = grid.detach().cpu().numpy()
            visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
    if batch_len == 5:
        image_save = batch.transpose(1, 4).contiguous()
        image_save = image_save.view(
            (image_save.size(0) * image_save.size(1), image_save.size(4), image_save.size(2), image_save.size(3)))
        grid = make_grid(image_save, nrow=nrow, padding=2, pad_value=1)
        save_img = grid.detach().cpu().numpy()
        visualize(np.transpose(save_img, (1, 2, 0)), join(dir, name))
