#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:32:01
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
""" Visualization utilities """
from __future__ import print_function, division, absolute_import, unicode_literals
from sys import version_info

PY3 = version_info[0] > 2

def svg2str(display_object, dpi=300):
    """
    Removes the preamble of the svg files generated with matplotlib
    """
    from io import StringIO
    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(image_buf, dpi=dpi, format='svg', facecolor='k',
                                             edgecolor='k')
    image_buf.seek(0)
    return image_buf.getvalue()

def extract_svg(display_object, dpi=300):
    """
    Removes the preamble of the svg files generated with matplotlib
    """
    from io import StringIO
    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format='svg',
        facecolor='k', edgecolor='k')
    image_buf.seek(0)
    image_svg = image_buf.getvalue()
    start_idx = image_svg.find('<svg ')
    end_idx = image_svg.rfind('</svg>')
    return image_svg[start_idx:end_idx]

def combine_svg(svg_list, out_file):
    """
    Composes the input svgs into one standalone svg
    """
    import numpy as np
    import svgutils.transform as svgt
    import svgutils.compose as svgc

    # Read all svg files and get roots
    svgs = [svgt.fromstring(f.encode('utf-8')) for f in svg_list]
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = [(int(f.width[:-2]), int(f.height[:-2])) for f in svgs]


    # Calculate the scale to fit all widths
    scales = [1.0] * len(svgs)
    if not all([width[0] == sizes[0][0] for width in sizes[1:]]):
        ref_size = sizes[0]
        for i, els in enumerate(sizes):
            scales[i] = ref_size[0]/els[0]

    newsizes = [tuple(size)
                for size in np.array(sizes) * np.array(scales)[..., np.newaxis]]

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    totalsize = [newsizes[0][0], np.sum(newsizes, axis=0)[1]]
    fig = svgt.SVGFigure(totalsize[0], totalsize[1])

    yoffset = 0
    for i, r in enumerate(roots):
        size = newsizes[i]
        r.moveto(0, yoffset, scale=scales[i])
        yoffset += size[1]
        fig.append(r)

    fig.save(out_file)
    return out_file



def combine_svg_verbose(
        in_brainmask,
        in_segmentation,
        in_artmask,
        in_headmask,
        in_airmask,
        in_bgplot):
    import os.path as op
    import svgutils.transform as svgt
    import svgutils.compose as svgc
    import numpy as np

    hspace = 10
    wspace = 10
    #create new SVG figure
    in_mosaics = [in_brainmask,
                  in_segmentation,
                  in_artmask,
                  in_headmask,
                  in_airmask]
    figs = [svgt.fromfile(f) for f in in_mosaics]

    roots = [f.getroot() for f in figs]
    nfigs = len(figs)

    sizes = [(int(f.width[:-2]), int(f.height[:-2])) for f in figs]
    maxsize = np.max(sizes, axis=0)
    minsize = np.min(sizes, axis=0)

    bgfile = svgt.fromfile(in_bgplot)
    bgscale = (maxsize[1] * 2 + hspace)/int(bgfile.height[:-2])
    bgsize = (int(bgfile.width[:-2]), int(bgfile.height[:-2]))
    bgfileroot = bgfile.getroot()

    totalsize = (minsize[0] + hspace + int(bgsize[0] * bgscale),
                 nfigs * maxsize[1] + (nfigs - 1) * hspace)
    fig = svgt.SVGFigure(svgc.Unit(totalsize[0]).to('cm'),
                         svgc.Unit(totalsize[1]).to('cm'))

    yoffset = 0
    for i, r in enumerate(roots):
        xoffset = 0
        if sizes[i][0] == maxsize[0]:
            xoffset = int(0.5 * (totalsize[0] - sizes[i][0]))
        r.moveto(xoffset, yoffset)
        yoffset += maxsize[1] + hspace

    bgfileroot.moveto(minsize[0] + wspace, 3 * (maxsize[1] + hspace), scale=bgscale)

    fig.append(roots + [bgfileroot])
    out_file = op.abspath('fig_final.svg')
    fig.save(out_file)
    return out_file
