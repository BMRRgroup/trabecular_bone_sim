import numpy as np
import scipy.ndimage as ndi
import utils
import os
import SimpleITK as SITK
import r2star
import pickle
from skimage.measure import block_reduce

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Ren Marrow fat spectrum
FREQS_PPM = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.950, -0.5, 0.49, 0.59])
REL_AMPS = np.array([0.0899, 0.5834, 0.0599, 0.0849, 0.0599, 0.0150, 0.04, 0.01, 0.0569])

# MR constant
GAMMABAR = 42.58e6  # [Hz/T]


class TraBoneSim(object):

    def __init__(self, bonecube_id='', inputdir='.', outputdir='.', field_strength_t=3, b0dir=[0, 0, 1], dchi_ppm=2,
                 te_s=[0.2e-3, 5e-3], rho_bone=0.3, t2_bone_ms=0.5, rho_marrow=1, t2_marrow_ms=60, fat_spec=False,
                 binary_erosion=0, binary_erosion_se=None, voxel_size_ct_mm=45.6e-3, voxel_size_mr_mm=1.5):
        """
        Class to simulate magnetic properties in MRI voxel
        containing trabecular bone

        :param bonecube_id: id of mask e.g. 'P8'
        :param inputdir: input directory
        :param outputdir: output directory
        :param field_strength_t: field strength in tesla
        :param b0dir: direction of main magentic field e.g. [0, 0, 1] for z axis
        :param dchi_ppm: susceptibility difference chi_bone - chi_marrow
        :param te_s: array of echo times in seconds
        :param rho_bone: density of bone
        :param t2_bone_ms: t2 of bone in ms
        :param rho_marrow: density of marrow
        :param t2_marrow_ms: t2 of marrow in ms
        :param fat_spec: use fat sepctrum [True, False]
        :param binary_erosion: use binary erosion to change bv/tv n: use erosion mask n times, 0/None no erosion, -n use dilation n times
        :param binary_erosion_se: binary erosion mask
        :param voxel_size_ct_mm: voxel size of micro resolution
        :param voxel_size_mr_mm: voxel size of macro resolution
        """
        self.smicro = None  # simulation result ct resolution
        self.smacro = None  # simulation result mr resolution
        self.bbox = None
        self.bbox_macro = None
        self.mask = None  # bone mask
        self.rdf_ppm = None  # relative distance field in ppm
        self.r2s_macro_hz = None  # r2* fit with simulated signal
        self.file_id = None

        self.bonecube_id = bonecube_id
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.field_strength_t = field_strength_t
        self.b0dir = b0dir
        self.dchi_ppm = dchi_ppm
        self.te_s = te_s
        self.rho_bone = rho_bone
        self.rho_marrow = rho_marrow
        self.t2_bone_ms = t2_bone_ms
        self.t2_marrow_ms = t2_marrow_ms
        self.fat_spec = fat_spec
        self.binary_erosion = binary_erosion
        self.binary_erosion_se = binary_erosion_se
        self.voxel_size_ct_mm = voxel_size_ct_mm
        self.voxel_size_mr_mm = voxel_size_mr_mm

    def set_bone_mask(self):
        """
        load CT data and create bone mask
        :return:
        """

        pstring = ('bonecube_id = {}\n'
                   'inputdir = {}\n'
                   'binary_erosion = {}\n'
                   'binary_erosion_se = {}\n').format(self.bonecube_id,
                                                      self.inputdir,
                                                      self.binary_erosion,
                                                      self.binary_erosion_se)

        print('\n' + pstring)
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        print('Load Data. ', end='')
        filename = os.path.join(self.inputdir, self.bonecube_id + '_mask.nii.gz')
        mask_image = SITK.ReadImage(filename)
        mask = np.moveaxis(SITK.GetArrayFromImage(mask_image), 0, -1)
        # erosion
        if self.binary_erosion is not None and (self.binary_erosion != 0):
            if self.binary_erosion is None:
                se = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                               [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                               [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
            else:
                se = self.binary_erosion_se

            if self.binary_erosion > 0:
                mask = ndi.binary_erosion(mask, iterations=self.binary_erosion, structure=se)
            elif self.binary_erosion < 0:
                mask = ndi.binary_dilation(mask, iterations=np.abs(self.binary_erosion), structure=se)

        # zero pad bone mask
        padwidth = np.max(mask.shape) // 2
        mask = np.pad(mask, (padwidth, padwidth), 'constant').astype('bool')

        # find bounding box
        self.bbox = tuple(ndi.find_objects(mask.astype('int'))[0])
        self.mask = mask

        print('Done.')

    def simulate_rdf_ppm(self):
        """
        forward simulation relative difference field (RDF)
        :return:
        """
        print('Forward simulate RDF. ', end='')

        chimap_ppm = self.dchi_ppm * self.mask
        rdf_ppm = utils.simulate_RDF_ppm(chimap_ppm, np.ones(3) * self.voxel_size_ct_mm, self.b0dir)

        print('Done.')
        self.rdf_ppm = rdf_ppm

    def simulate_signal(self):
        """
        signal generation
        :return:
        """

        # generate (microscopic) MR signal
        print('Generate microscopic signal. ', end='')

        # set center frequency
        center_freq_hz = GAMMABAR * self.field_strength_t

        # tissue constants
        r2_bone_hz = 1000 / self.t2_bone_ms
        r2_marrow_hz = 1000 / self.t2_marrow_ms

        # crop white space and get quantitative maps
        fieldmap_hz = center_freq_hz * self.rdf_ppm[self.bbox] * 1e-6

        one_d_bone_mask = np.ndarray.flatten(self.mask[self.bbox])
        one_d_marrow_mask = np.logical_not(one_d_bone_mask)
        one_d_fieldmap_hz = np.ndarray.flatten(fieldmap_hz)

        n_te = len(self.te_s)
        size = np.array([*one_d_fieldmap_hz.shape, n_te])

        s = np.zeros(size, dtype='complex')

        if self.fat_spec:

            freqs_hz = center_freq_hz * FREQS_PPM * 1e-6

            for (i, t) in enumerate(self.te_s):
                shift_hz_marrow = (np.sum(REL_AMPS * np.exp(2j * np.pi * freqs_hz * t)))

                s[one_d_marrow_mask, i] = self.rho_marrow * np.exp(
                    2j * np.pi * one_d_fieldmap_hz[one_d_marrow_mask] * t) * np.exp(-t * r2_marrow_hz) * shift_hz_marrow

                s[one_d_bone_mask, i] = self.rho_bone * np.exp(
                    2j * np.pi * one_d_fieldmap_hz[one_d_bone_mask] * t) * np.exp(-t * r2_bone_hz)

        else:
            for (i, t) in enumerate(self.te_s):
                s[one_d_marrow_mask, i] = self.rho_marrow * np.exp(
                    2j * np.pi * one_d_fieldmap_hz[one_d_marrow_mask] * t) * np.exp(-t * r2_marrow_hz)

                s[one_d_bone_mask, i] = self.rho_bone * np.exp(
                    2j * np.pi * one_d_fieldmap_hz[one_d_bone_mask] * t) * np.exp(-t * r2_bone_hz)

        print('Done.')
        self.smicro = np.reshape(s, np.array([*fieldmap_hz.shape, n_te]))

    def downsample_signal(self):
        """
        downsample to "macro" MR resolution ~1.5mm isotropic
        :return:
        """

        print('Downsample to MR signal. ', end='')
        downsample_factor = np.floor(self.voxel_size_mr_mm / self.voxel_size_ct_mm)
        d = int(downsample_factor / 2)
        size = np.array(self.smicro.shape)
        newsize = np.append(np.ceil(size[0:3] / d).astype(int), size[3])
        s = np.zeros(newsize, dtype='complex')
        for i in range(size[3]):
            s[:, :, :, i] = block_reduce(self.smicro[:, :, :, i],
                                               block_size=(d, d, d), func=np.mean)
        print('Done.')

        self.smacro = s

    def set_bbox_macro(self):
        """
        approximate bounding box
        :return:
        """

        inds0 = np.array(self.smacro.shape[:3]) // 4
        inds1 = 3 * np.array(self.smacro.shape[:3]) // 4
        self.bbox_macro = [slice(inds0[0], inds1[0]),
                           slice(inds0[1], inds1[1]),
                           slice(inds0[2], inds1[2])]

    def fit_r2s_macro(self):
        """
        R2* fitting of simulated mr signal
        :return:
        """
        print('Fit R2*. ', end='')
        self.r2s_macro_hz = r2star.fit_monoexpdecay_gss(self.te_s, np.abs(self.smacro), 0, 1000)
        print('Done.')

    def get_fieldmap_macro(self):
        """
        simple field mapping
        :return:
        """

        print('Field mapping. ', end='')
        iTE0 = 0
        iTE1 = -1
        centerFreq_Hz = GAMMABAR * self.field_strength_t
        self.rdf_ppm = 1e6 * (np.angle(self.smacro[:, :, :, iTE1]) - np.angle(self.smacro[:, :, :, iTE0])) / \
                       (2 * np.pi * centerFreq_Hz * (self.te_s[iTE1] - self.te_s[iTE0]))
        print('Done.')

    def plot(self, attr=['mask', 'RDF_ppm']):
        if isinstance(attr, list):
            for a in attr:
                iz = vars(self)[a].shape[2] // 2

                fig, ax = plt.subplots(figsize=(15, 6))
                im = ax.imshow(vars(self)[a][:, :, iz], cmap='inferno')
                ax.set_axis_off()

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)

                plt.show()

    def get_bvtv(self):
        """
        bone volume / total volume (BV/TV)
        """
        return np.mean(np.abs(self.mask[self.bbox].astype('int')))

    def set_file_id(self):
        """
        set file id
        :return:
        """
        has_bone_signal = False

        if self.rho_bone > 0:
            has_bone_signal = True

        file_id = self.bonecube_id + '_b0dir{}{}{}'.format(*self.b0dir)
        file_id = file_id + '_withBone' if has_bone_signal == True else file_id + '_noBone'

        if self.binary_erosion is not None and (self.binary_erosion != 0):
            if self.binary_erosion > 0:
                file_id = file_id + '_bE' + str(self.binary_erosion)
            elif self.binary_erosion < 0:
                file_id = file_id + '_bD' + str(np.abs(self.binary_erosion))

        if self.fat_spec:
            file_id = file_id + '_withFatSpec'

        self.file_id = file_id

    def save_file(self):
        """
        save file as npz
        :return:
        """

        self.set_file_id()

        pstring = ('bonecube_id = {}\n'
                   'inputdir = {}\n'
                   'outputdir = {}\n'
                   'field_strength_t = {}\n'
                   'b0dir = {}\n'
                   'dchi_ppm = {}\n'
                   'te_s = {}\n'
                   't2_bone_ms = {}\n'
                   't2_marrow_ms = {}\n'
                   'rho_bone = {}\n'
                   'binary_erosion_se =\n{}\n'
                   'useFatSpec = {}\n').format(self.bonecube_id,
                                               self.inputdir,
                                               self.outputdir,
                                               self.field_strength_t,
                                               self.b0dir,
                                               self.dchi_ppm,
                                               self.te_s,
                                               self.t2_bone_ms,
                                               self.t2_marrow_ms,
                                               self.rho_bone,
                                               self.binary_erosion_se,
                                               self.fat_spec)

        # save simulation parameters
        with open(os.path.join(self.outputdir, self.file_id + '_sim_params.txt'), "w") as f:
            f.write(pstring)

        # save label statistics
        with open(os.path.join(self.outputdir, self.file_id + '_sim_stats.csv'), "w") as f:
            f.write(
                'mean_BVTV, var_BVTV, mean_R2s_Hz, var_R2s, min_R2s, max_R2s, mean_chi_ppm, var_chi, min_chi, max_chi\n')
            f.write('{},{},{},{},{},{}\n'.format(self.mask[self.bbox].mean(),
                                                 self.mask[self.bbox].var(),
                                                 self.r2s_macro_hz[self.bbox].mean(),
                                                 self.r2s_macro_hz[self.bbox].var(),
                                                 self.r2s_macro_hz[self.bbox].min(),
                                                 self.r2s_macro_hz[self.bbox].max()))

        filename = os.path.join(self.outputdir, self.file_id)

        print('Save file:' + filename)
        np.savez(filename, smicro=self.smicro, smacro=self.smacro, mask=self.mask, RDF_ppm=self.rdf_ppm,
                 pstring=pstring)
        print('Done.')

    def save_object(self):
        self.set_file_id()
        pickle.dump(self, self.file_id, pickle.HIGHEST_PROTOCOL)


