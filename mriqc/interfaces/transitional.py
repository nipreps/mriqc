#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipype.interfaces.base import (
    File, traits, CommandLine, TraitedSpec, CommandLineInputSpec
)


class GCORInputSpec(CommandLineInputSpec):
    in_file = File(
        desc='input dataset to compute the GCOR over',
        argstr='-input %s',
        position=-1,
        mandatory=True,
        exists=True,
        copyfile=False)

    mask = File(
        desc='mask dataset, for restricting the computation',
        argstr='-mask %s',
        exists=True,
        copyfile=False)

    nfirst = traits.Int(0, argstr='-nfirst %d',
                        desc='specify number of initial TRs to ignore')
    no_demean = traits.Bool(False, argstr='-no_demean',
                            desc='do not (need to) demean as first step')


class GCOROutputSpec(TraitedSpec):
    out = traits.Float(desc='global correlation value')


class GCOR(CommandLine):
    """
    Computes the average correlation between every voxel
    and ever other voxel, over any give mask.
    For complete details, see the `@compute_gcor Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/@compute_gcor.html>`_

    Examples
    ========
    >>> from mriqc.interfaces.transitional import GCOR
    >>> gcor = GCOR()
    >>> gcor.inputs.in_file = 'func.nii'
    >>> gcor.inputs.nfirst = 4
    >>> gcor.cmdline  # doctest: +ALLOW_UNICODE
    '@compute_gcor -nfirst 4 -input func.nii'
    >>> res = gcor.run()  # doctest: +SKIP

    """

    _cmd = '@compute_gcor'
    input_spec = GCORInputSpec
    output_spec = GCOROutputSpec

    def _run_interface(self, runtime):
        runtime = super(GCOR, self)._run_interface(runtime)

        gcor_line = [line.strip() for line in runtime.stdout.split('\n')
                     if line.strip().startswith('GCOR = ')][-1]
        setattr(self, '_gcor', float(gcor_line[len('GCOR = '):]))
        return runtime

    def _list_outputs(self):
        return {'out': getattr(self, '_gcor')}
