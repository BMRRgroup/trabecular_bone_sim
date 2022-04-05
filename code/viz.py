import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_voxelSignal(s, ind):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.arange(1, s.shape[-1]+1), np.abs(s[ind[0], ind[1], ind[2], :]), 'o-')
    ax[1].plot(np.arange(1, s.shape[-1]+1), np.angle(s[ind[0], ind[1], ind[2], :]), 'o-')
    # plt.show()


def slicer(arr):
    os.environ["SITK_SHOW_COMMAND"] = "/Applications/Slicer.app/Contents/MacOS/Slicer"
    sitk.Show(sitk.GetImageFromArray(np.moveaxis(arr, -1, 0)))


def imagej(arr):
    os.environ["SITK_SHOW_COMMAND"] = "/Applications/ImageJ/ImageJ.app/Contents/MacOS/JavaApplicationStub"
    sitk.Show(sitk.GetImageFromArray(np.moveaxis(arr, -1, 0)))


def itksnap(arr):
    os.environ["SITK_SHOW_COMMAND"] = "/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP"
    sitk.Show(sitk.GetImageFromArray(np.moveaxis(arr, -1, 0)))


def plot_slice(arr, iz, **kwargs):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(arr[:, :, iz], **kwargs)
    plt.colorbar(im)
    ax.grid(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # plt.show()


def plot2d(aslice, **kwargs):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(aslice, **kwargs)
    ax.grid(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # plt.show()


def plot_orientations(img, **kwargs):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 6))
    ix = img.shape[0] // 2
    iy = img.shape[1] // 2
    iz = img.shape[2] // 2
    im = axs[0].imshow(img[:, :, iz], **kwargs)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("left", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position('left')
    axs[1].imshow(img[:, iy, :], **kwargs)
    axs[2].imshow(img[ix, :, :], **kwargs)
    axs[0].set_title('XY')
    axs[1].set_title('YZ')
    axs[2].set_title('XZ')
    for ax in axs:
        ax.grid(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    return fig, axs
    # plt.show()
