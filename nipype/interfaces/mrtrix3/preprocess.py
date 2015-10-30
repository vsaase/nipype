# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# -*- coding: utf-8 -*-

"""
    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname(os.path.realpath(__file__ ))
    >>> datadir = os.path.realpath(os.path.join(filepath,
    ...                            '../../testing/data'))
    >>> os.chdir(datadir)

"""
import os
import os.path as op

from .base import MRTrix3BaseInputSpec, MRTrix3Base
from ..base import (CommandLineInputSpec, CommandLine, traits, TraitedSpec,
                    File)
from ..traits_extension import isdefined
from ...utils.filemanip import split_filename


class MRConvertInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
        desc='voxel-order data filename')
    out_filename = File(genfile=True, argstr='%s', position=-1, desc='Output filename')
    extract_at_axis = traits.Enum(1,2,3, argstr='-coord %s', position=1,
                           desc='"Extract data only at the coordinates specified. This option specifies the Axis. Must be used in conjunction with extract_at_coordinate.')
    extract_at_coordinate = traits.List(traits.Float, argstr='%s', sep=',', position=2, minlen=1, maxlen=3,
        desc='"Extract data only at the coordinates specified. This option specifies the coordinates. Must be used in conjunction with extract_at_axis. Three comma-separated numbers giving the size of each voxel in mm.')
    voxel_dims = traits.List(traits.Float, argstr='-vox %s', sep=',',
        position=3, minlen=3, maxlen=3,
        desc='Three comma-separated numbers giving the size of each voxel in mm.')
    output_datatype = traits.Enum("nii", "float", "char", "short", "int", "long", "double", argstr='-datatype %s', position=2,
                           desc='"i.e. Bfloat". Can be "char", "short", "int", "long", "float" or "double"') #, usedefault=True)
    extension = traits.Enum("mif","nii", "float", "char", "short", "int", "long", "double", position=2,
                           desc='"i.e. Bfloat". Can be "char", "short", "int", "long", "float" or "double"', usedefault=True)
    layout = traits.Enum("nii", "float", "char", "short", "int", "long", "double", argstr='-output %s', position=2,
                           desc='specify the layout of the data in memory. The actual layout produced will depend on whether the output image format can support it.')
    scaling = traits.Float(argstr='-scaling %d', position=3,
        units='mm', desc='''specify the data scaling parameters used to rescale the intensity values.
     These take the form of a comma-separated 2-vector of floating-point
     values, corresponding to offset & scale, with final intensity values being
     given by offset + scale * stored_value. By default, the values in the
     input image header are passed through to the output image header when
     writing to an integer image, and reset to 0,1 (no scaling) for
     floating-point and binary images. Note that his option has no effect for
     floating-point and binary images.''')
        
class MRConvertOutputSpec(TraitedSpec):
    converted = File(exists=True, desc='path/name of 4D volume in voxel order')

class MRConvert(MRTrix3Base):
    """
    Perform conversion between different file types and optionally extract a subset of the input image.

    If used correctly, this program can be a very useful workhorse.
    In addition to converting images between different formats, it can
    be used to extract specific studies from a data set, extract a specific
    region of interest, flip the images, or to scale the intensity of the images.

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> mrconvert = mrt.MRConvert()
    >>> mrconvert.inputs.in_file = 'dwi_FA.mif'
    >>> mrconvert.inputs.out_filename = 'dwi_FA.nii'
    >>> mrconvert.run()                                 # doctest: +SKIP
    """

    _cmd = 'mrconvert'
    input_spec=MRConvertInputSpec
    output_spec=MRConvertOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['converted'] = self.inputs.out_filename
        if not isdefined(outputs['converted']):
            outputs['converted'] = op.abspath(self._gen_outfilename())
        else:
            outputs['converted'] = op.abspath(outputs['converted'])
        return outputs

    def _gen_filename(self, name):
        if name is 'out_filename':
            return self._gen_outfilename()
        else:
            return None
    def _gen_outfilename(self):
        _, name , _ = split_filename(self.inputs.in_file)
        if isdefined(self.inputs.out_filename):
            outname = self.inputs.out_filename
        else:
            outname = name + '_mrconvert.' + self.inputs.extension
        return outname


class ResponseSDInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
                   desc='input diffusion weighted images')

    out_file = File(
        'response.txt', argstr='%s', mandatory=True, position=-1,
        usedefault=True, desc='output file containing SH coefficients')

    # DW Shell selection options
    shell = traits.List(traits.Float, sep=',', argstr='-shell %s',
                        desc='specify one or more dw gradient shells')
    in_mask = File(exists=True, argstr='-mask %s',
                   desc='provide initial mask image')
    max_sh = traits.Int(8, argstr='-lmax %d',
                        desc='maximum harmonic degree of response function')
    out_sf = File('sf_mask.nii.gz', argstr='-sf %s',
                  desc='write a mask containing single-fibre voxels')
    test_all = traits.Bool(False, argstr='-test_all',
                           desc='re-test all voxels at every iteration')

    # Optimization
    iterations = traits.Int(0, argstr='-max_iters %d',
                            desc='maximum number of iterations per pass')
    max_change = traits.Float(
        argstr='-max_change %f',
        desc=('maximum percentile change in any response function coefficient;'
              ' if no individual coefficient changes by more than this '
              'fraction, the algorithm is terminated.'))

    # Thresholds
    vol_ratio = traits.Float(
        .15, argstr='-volume_ratio %f',
        desc=('maximal volume ratio between the sum of all other positive'
              ' lobes in the voxel and the largest FOD lobe'))
    disp_mult = traits.Float(
        1., argstr='-dispersion_multiplier %f',
        desc=('dispersion of FOD lobe must not exceed some threshold as '
              'determined by this multiplier and the FOD dispersion in other '
              'single-fibre voxels. The threshold is: (mean + (multiplier * '
              '(mean - min))); default = 1.0. Criterion is only applied in '
              'second pass of RF estimation.'))
    int_mult = traits.Float(
        2., argstr='-integral_multiplier %f',
        desc=('integral of FOD lobe must not be outside some range as '
              'determined by this multiplier and FOD lobe integral in other'
              ' single-fibre voxels. The range is: (mean +- (multiplier * '
              'stdev)); default = 2.0. Criterion is only applied in second '
              'pass of RF estimation.'))


class ResponseSDOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')
    out_sf = File(desc=('mask containing single-fibre voxels'))


class ResponseSD(MRTrix3Base):

    """
    Generate an appropriate response function from the image data for
    spherical deconvolution.

    .. [1] Tax, C. M.; Jeurissen, B.; Vos, S. B.; Viergever, M. A. and
      Leemans, A., Recursive calibration of the fiber response function
      for spherical deconvolution of diffusion MRI data. NeuroImage,
      2014, 86, 67-80


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> resp = mrt.ResponseSD()
    >>> resp.inputs.in_file = 'dwi.mif'
    >>> resp.inputs.in_mask = 'mask.nii.gz'
    >>> resp.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> resp.cmdline                               # doctest: +ELLIPSIS
    'dwi2response -fslgrad bvecs bvals -mask mask.nii.gz dwi.mif response.txt'
    >>> resp.run()                                 # doctest: +SKIP
    """

    _cmd = 'dwi2response'
    input_spec = ResponseSDInputSpec
    output_spec = ResponseSDOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)

        if isdefined(self.inputs.out_sf):
            outputs['out_sf'] = op.abspath(self.inputs.out_sf)
        return outputs


class ACTPrepareFSLInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
                   desc='input anatomical image')

    out_file = File(
        'act_5tt.nii.gz', argstr='%s', mandatory=True, position=-1,
        usedefault=True, desc='output file after processing')


class ACTPrepareFSLOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')
    out_brain = File(exists=True, desc='the brain image for registration purposes')


class ACTPrepareFSL(CommandLine):

    """
    Generate anatomical information necessary for Anatomically
    Constrained Tractography (ACT).

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> prep = mrt.ACTPrepareFSL()
    >>> prep.inputs.in_file = 'T1.nii.gz'
    >>> prep.cmdline                               # doctest: +ELLIPSIS
    'act_anat_prepare_fsl T1.nii.gz act_5tt.mif'
    >>> prep.run()                                 # doctest: +SKIP
    """

    _cmd = 'act_anat_prepare_fsl -tempdir temp -nocleanup'
    input_spec = ACTPrepareFSLInputSpec
    output_spec = ACTPrepareFSLOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        outputs['out_brain'] = os.path.abspath('temp/T1_bet.nii.gz')
        return outputs

class MakeGMWMIInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
                   desc='input 5TT image')

    out_file = File(
        'gmwmi.nii.gz', argstr='%s', mandatory=True, position=-1,
        usedefault=True, desc='output file after processing')


class MakeGMWMIOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')


class MakeGMWMI(CommandLine):

    """
    Generate GM / WM Interface mask for Anatomically
    Constrained Tractography (ACT).
    """

    _cmd = '5tt2gmwmi'
    input_spec = MakeGMWMIInputSpec
    output_spec = MakeGMWMIOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs


class ReplaceFSwithFIRSTInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-4,
                   desc='input anatomical image')
    in_t1w = File(exists=True, argstr='%s', mandatory=True, position=-3,
                  desc='input T1 image')
    in_config = File(exists=True, argstr='%s', position=-2,
                     desc='connectome configuration file')

    out_file = File(
        'aparc+first.mif', argstr='%s', mandatory=True, position=-1,
        usedefault=True, desc='output file after processing')


class ReplaceFSwithFIRSTOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output response file')


class ReplaceFSwithFIRST(CommandLine):

    """
    Replace deep gray matter structures segmented with FSL FIRST in a
    FreeSurfer parcellation.

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> prep = mrt.ReplaceFSwithFIRST()
    >>> prep.inputs.in_file = 'aparc+aseg.nii'
    >>> prep.inputs.in_t1w = 'T1.nii.gz'
    >>> prep.inputs.in_config = 'mrtrix3_labelconfig.txt'
    >>> prep.cmdline                               # doctest: +ELLIPSIS
    'fs_parc_replace_sgm_first aparc+aseg.nii T1.nii.gz \
mrtrix3_labelconfig.txt aparc+first.mif'
    >>> prep.run()                                 # doctest: +SKIP
    """

    _cmd = 'fs_parc_replace_sgm_first'
    input_spec = ReplaceFSwithFIRSTInputSpec
    output_spec = ReplaceFSwithFIRSTOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs
