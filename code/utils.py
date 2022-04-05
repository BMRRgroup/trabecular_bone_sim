import os
import numpy as np
from numpy.fft import fftn, ifftn, fftshift

import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
from viz import plot_orientations

def close_all(viewer=['JavaApplicationStub', 'ITK-SNAP']):
    for v in viewer:
        kill_processes(v)

def killITK():
    kill_processes('ITK-SNAP')


def killIJ():
    kill_processes('JavaApplicationStub')


def kill_processes(procname):
    import psutil
    for proc in psutil.process_iter():
        pinfo = proc.as_dict(attrs=['pid', 'name'])
        if procname in pinfo['name']:
            proc.kill()
            print('killed:', pinfo)


def show_arr3d(arr, tool='ITK-SNAP'):
    SITK_SHOW_COMMAND = os.environ.get('SITK_SHOW_COMMAND')

    # setup
    if tool == 'ITK-SNAP':
        os.environ['SITK_SHOW_COMMAND'] = \
            '/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP'
    elif tool == 'imagej':
        os.environ['SITK_SHOW_COMMAND'] = \
            '/Applications/ImageJ/ImageJ.app/Contents/MacOS/JavaApplicationStub'
    elif tool == 'slicer':
        os.environ['SITK_SHOW_COMMAND'] = \
            '/Applications/Slicer.app/Contents/MacOS/Slicer'
    else:
        os.environ['SITK_SHOW_COMMAND'] = tool

    # SimpleITK takes array's in scikit.ndimage convention [z, y, x] and
    # ITK-snap flips i and j directions
    # therefore transpose array to [k, j, i] and flip
    # and then flip i, j to [k, i, j] => transpose order [2, 0, 1]
    # sitk.Show(sitk.GetImageFromArray(
    #     np.transpose(arr.astype(float), [2, 0, 1])))
    sitk.Show(sitk.GetImageFromArray(np.transpose(arr.astype(float))))
    print()

    # teardown
    if SITK_SHOW_COMMAND is not None:
        os.environ['SITK_SHOW_COMMAND'] = SITK_SHOW_COMMAND

def dipole_inversion_closedForm_l2(rdf_ppm, matrixSize, voxelSize_mm,
                                   b0dir, reg):
    """
    Implementation of closed form dipole inversion solution
    Bilgic et al., Fast image reconstruction with l2-regularization, JMRI 2013
    http://dx.doi.org/10.1002/jmri.24365
    """
    D = get_dipoleKernel_kspace(matrixSize, voxelSize_mm, b0dir)
    D2 = np.abs(D)**2

    Ei, Ej, Ek = get_grad_kspace(matrixSize)
    E2 = np.abs(Ei)**2 + np.abs(Ej)**2 + np.abs(Ek)**2

    A = fftshift(D2) + reg * fftshift(E2)
    b = fftshift(D) * fftn(rdf_ppm)
    x = b / (A + np.spacing(1))
    chi_ppm = np.real(ifftn(x))
    return chi_ppm


def get_dipoleKernel_kspace(matrixSize, voxelSize_mm, b0dir, DCoffset=0):
    """
    dipole kernel in kspace D = 1/3 - kz^2 / k^2

    see e.g.
    Marques & Bowtell, Application of a fourier-based method for
    rapid calculation of field inhomogeneity due to spatial
    variation of magnetic susceptibility,
    http://dx.doi.org/10.1002/cmr.b.20034
    """
    i = np.linspace(-matrixSize[0]//2, matrixSize[0]//2 - 1, matrixSize[0])
    j = np.linspace(-matrixSize[1]//2, matrixSize[1]//2 - 1, matrixSize[1])
    k = np.linspace(-matrixSize[2]//2, matrixSize[2]//2 - 1, matrixSize[2])
    J, I, K = np.meshgrid(j, i, k)

    FOV_mm = voxelSize_mm * matrixSize
    dk = 1 / FOV_mm
    Ki = dk[0] * I
    Kj = dk[1] * J
    Kk = dk[2] * K

    Kz = b0dir[0] * Ki + b0dir[1] * Kj + b0dir[2] * Kk
    K2 = Ki**2 + Kj**2 + Kk**2
    center = K2 == 0
    K2[center] = np.inf

    D = 1/3 - Kz**2 / K2
    D[center] = DCoffset
    return D


def get_grad_kspace(matrixSize):
    """
    forward finite difference operators in kspace
    """
    i = np.linspace(-matrixSize[0]//2, matrixSize[0]//2 - 1, matrixSize[0])
    j = np.linspace(-matrixSize[1]//2, matrixSize[1]//2 - 1, matrixSize[1])
    k = np.linspace(-matrixSize[2]//2, matrixSize[2]//2 - 1, matrixSize[2])
    J, I, K = np.meshgrid(j/matrixSize[1], i/matrixSize[0], k/matrixSize[2])

    Ei = 1 - np.exp(2j * np.pi * I)
    Ej = 1 - np.exp(2j * np.pi * J)
    Ek = 1 - np.exp(2j * np.pi * K)
    return Ei, Ej, Ek


def apply_kspaceFilter(arr, kspaceFilter_fftshifted):
    return np.real(ifftn(kspaceFilter_fftshifted * fftn(arr)))


def simulate_RDF_ppm(chi_ppm, voxelSize_mm, b0dir):
    """
    simple way to forward simulate the relative difference field (RDF_ppm)
    no zero padding done here -> circular convolution no problem
    because bone cube is well inside the FOV with a large boarder of 0s

    see e.g.
    Marques & Bowtell, Application of a fourier-based method for
    rapid calculation of field inhomogeneity due to spatial
    variation of magnetic susceptibility,
    http://dx.doi.org/10.1002/cmr.b.20034
    """
    matrixSize = chi_ppm.shape
    D = get_dipoleKernel_kspace(matrixSize, voxelSize_mm, b0dir)
    return apply_kspaceFilter(chi_ppm, fftshift(D))



def load_npz(filename, names=['S', ]):
    print('Load file:' + filename)
    with np.load(filename + '.npz') as data:
        S = data['Smicro']
        s = data['Smacro']
        mask = data['mask']
        R2s_Hz = data['R2s_Hz']
        RDF_ppm = data['RDF_ppm']
        rdf_ppm = data['rdf_ppm']
        print('Done.')
    return S, s, mask, RDF_ppm, rdf_ppm, R2s_Hz


def save_3Darray(arr, filename):
    sitk.WriteImage(
        sitk.GetImageFromArray(np.transpose(arr)),
        filename)
    print('Wrote {}.'.format(filename))


def save_4Darray(arr, filename):
    nii = nib.Nifti1Image(arr, None)
    nib.save(nii, filename)
    print('Wrote {}.'.format(filename))


def save_niftis(S, s, mask, RDF_ppm, rdf_ppm, R2s_Hz,
                chi_ppm, pstring, outputdir, fileID):
    print('Save niftis.')
    save_4Darray(S, os.path.join(outputdir, fileID+'_signal.nii.gz'))
    save_4Darray(s, os.path.join(outputdir, fileID+'_signal_downsamp.nii.gz'))
    save_3Darray(mask.astype('float'),
                 os.path.join(outputdir, fileID+'_mask.nii.gz'))
    save_3Darray(R2s_Hz, os.path.join(outputdir, fileID+'_R2s.nii.gz'))
    save_3Darray(RDF_ppm, os.path.join(outputdir, fileID+'_RDF.nii.gz'))
    save_3Darray(rdf_ppm,
                 os.path.join(outputdir, fileID+'_RDF_downsamp.nii.gz'))
    save_3Darray(chi_ppm, os.path.join(outputdir, fileID+'_chi.nii.gz'))
    print('Done.')


def save_figures(S, s, mask, RDF_ppm, rdf_ppm, R2s_Hz,
                 chi_ppm, pstring, outputdir, fileID):
    print('Save plots. ', end='')
    nTE = S.shape[-1]
    fig, axs = plt.subplots(2, nTE, figsize=(25, 10))
    iz = S.shape[2] // 2
    axs[0][0].set_ylabel('magnitude')
    axs[1][0].set_ylabel('phase')
    for i in range(nTE):
        axs[0][i].imshow(np.abs(S[:, :, iz, i]), vmin=0, vmax=1)
        axs[0][i].xaxis.set_ticks([])
        axs[0][i].yaxis.set_ticks([])
        axs[1][i].imshow(np.angle(S[:, :, iz, i]), vmin=-np.pi, vmax=np.pi)
        axs[1][i].set_xlabel('TE {}'.format(i + 1))
        axs[1][i].xaxis.set_ticks([])
        axs[1][i].yaxis.set_ticks([])
    plt.suptitle(fileID + ' signal')
    plt.tight_layout()
    # tikz_save(os.path.join(outputdir, fileID + '_signal.tex'))
    plt.savefig(os.path.join(outputdir, fileID + '_signal.png'))
    plt.savefig(os.path.join(outputdir, fileID + '_signal.pdf'))

    nTE = s.shape[-1]
    fig, axs = plt.subplots(2, nTE, figsize=(25, 10))
    iz = s.shape[2] // 2
    axs[0][0].set_ylabel('magnitude')
    axs[1][0].set_ylabel('phase')
    for i in range(nTE):
        axs[0][i].imshow(np.abs(s[:, :, iz, i]), vmin=0, vmax=1)
        axs[0][i].xaxis.set_ticks([])
        axs[0][i].yaxis.set_ticks([])
        axs[1][i].imshow(np.angle(s[:, :, iz, i]), vmin=-np.pi, vmax=np.pi)
        axs[1][i].set_xlabel('TE {}'.format(i + 1))
        axs[1][i].xaxis.set_ticks([])
        axs[1][i].yaxis.set_ticks([])
    axs[1][-1].set_xlabel('TE {}'.format(i + 1))
    axs[1][-1].xaxis.set_ticks([])
    axs[1][-1].yaxis.set_ticks([])
    plt.suptitle(fileID + ' signal downsampled')
    plt.tight_layout()
    # tikz_save(os.path.join(outputdir, fileID + '_signal_downsamp.tex'))
    plt.savefig(os.path.join(outputdir, fileID + '_signal_downsamp.png'))
    plt.savefig(os.path.join(outputdir, fileID + '_signal_downsamp.pdf'))

    plot_orientations(mask, cmap='viridis', vmin=0, vmax=1)
    plt.suptitle(fileID+' mask')
    # tikz_save(os.path.join(outputdir, fileID + '_mask.tex'))
    plt.savefig(os.path.join(outputdir, fileID + '_mask.png'))
    plt.savefig(os.path.join(outputdir, fileID + '_mask.pdf'))

    plot_orientations(RDF_ppm, cmap='inferno')
    plt.suptitle(fileID+' RDF [ppm]')
    # tikz_save(os.path.join(outputdir, fileID + '_RDF.tex'))
    plt.savefig(os.path.join(outputdir, fileID + '_RDF.png'))
    plt.savefig(os.path.join(outputdir, fileID + '_RDF.pdf'))

    plot_orientations(rdf_ppm, cmap='inferno')
    plt.suptitle(fileID+' RDF [ppm]')
    # tikz_save(os.path.join(outputdir, fileID + '_RDF_downsamp.tex'))
    plt.savefig(os.path.join(outputdir, fileID + '_RDF_downsamp.png'))
    plt.savefig(os.path.join(outputdir, fileID + '_RDF_downsamp.pdf'))

    plot_orientations(R2s_Hz, cmap='viridis', vmin=0, vmax=200)
    plt.suptitle(fileID+' R2* [s^-1]')
    # tikz_save(os.path.join(outputdir, fileID + '_R2s.tex'))
    plt.savefig(os.path.join(outputdir, fileID + '_R2s.png'))
    plt.savefig(os.path.join(outputdir, fileID + '_R2s.pdf'))

    plot_orientations(chi_ppm, cmap='viridis')
    plt.suptitle(fileID+' chi [ppm]')
    # tikz_save(os.path.join(outputdir, fileID + '_chi.tex'))
    plt.savefig(os.path.join(outputdir, fileID + '_chi.png'))
    plt.savefig(os.path.join(outputdir, fileID + '_chi.pdf'))

    plt.close('all')
    print('Done.')

def plot_signal(te_s, signal, fileID, fout):
    sz = np.array(signal.shape) // 2
    step = np.array(signal.shape) // 8
    inds0 = sz - step
    inds1 = sz + step
    box = [slice(inds0[0], inds1[0]),
           slice(inds0[1], inds1[1]),
           slice(inds0[2], inds1[2])]
    # average signal in ROI
    s = np.zeros(signal.shape[-1], dtype='complex')
    for iTE in range(signal.shape[-1]):
        ind = [*box, iTE]
        s[iTE] = signal[ind].mean()

    # create plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(fileID)
    ax[0].plot(te_s, abs(s), 'o-')
    ax[0].set_title('magnitude')
    ax[0].set_xlabel('TE [s]')
    ax[0].set_ylabel('[a.u.]')
    ax[0].grid(True)
    ax[1].plot(te_s, np.angle(s), 'o-')
    ax[1].set_title('phase')
    ax[1].set_xlabel('TE [s]')
    ax[1].set_ylabel('[rad]')
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(fout)
    print('Wrote {}.'.format(fout))

def plot_signal_single_voxel(voxel, te_s, signal, fileID, fout):
    s = []
    for iTE in range(signal.shape[-1]):
        s[iTE] = signal[voxel]

    # create plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(fileID)
    ax[0].plot(te_s, abs(s), 'o-')
    ax[0].set_title('magnitude')
    ax[0].set_xlabel('TE [s]')
    ax[0].set_ylabel('[a.u.]')
    ax[0].grid(True)
    ax[1].plot(te_s, np.angle(s), 'o-')
    ax[1].set_title('phase')
    ax[1].set_xlabel('TE [s]')
    ax[1].set_ylabel('[rad]')
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(fout)
    print('Wrote {}.'.format(fout))
