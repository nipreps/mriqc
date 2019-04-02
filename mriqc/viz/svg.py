#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:32:01
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
""" SVG handling utilities """


def svg2str(display_object, dpi=300):
    """
    Serializes a nilearn display object as a string
    """
    from io import StringIO
    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format='svg',
        facecolor='k', edgecolor='k')
    image_buf.seek(0)
    return image_buf.getvalue()


def combine_svg(svg_list, axis='vertical'):
    """
    Composes the input svgs into one standalone svg
    """
    import numpy as np
    import svgutils.transform as svgt

    # Read all svg files and get roots
    svgs = [svgt.fromstring(f.encode('utf-8')) for f in svg_list]
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = [(int(f.width[:-2]), int(f.height[:-2])) for f in svgs]

    if axis == 'vertical':
        # Calculate the scale to fit all widths
        scales = [1.0] * len(svgs)
        if not all([width[0] == sizes[0][0] for width in sizes[1:]]):
            ref_size = sizes[0]
            for i, els in enumerate(sizes):
                scales[i] = ref_size[0] / els[0]

        newsizes = [tuple(size)
                    for size in np.array(sizes) * np.array(scales)[..., np.newaxis]]
        totalsize = [newsizes[0][0], np.sum(newsizes, axis=0)[1]]

    elif axis == 'horizontal':
        # Calculate the scale to fit all heights
        scales = [1.0] * len(svgs)
        if not all([height[0] == sizes[0][1] for height in sizes[1:]]):
            ref_size = sizes[0]
            for i, els in enumerate(sizes):
                scales[i] = ref_size[1] / els[1]

        newsizes = [tuple(size)
                    for size in np.array(sizes) * np.array(scales)[..., np.newaxis]]
        totalsize = [np.sum(newsizes, axis=0)[0], newsizes[0][1]]

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = svgt.SVGFigure(totalsize[0], totalsize[1])

    if axis == 'vertical':
        yoffset = 0
        for i, r in enumerate(roots):
            size = newsizes[i]
            r.moveto(0, yoffset, scale=scales[i])
            yoffset += size[1]
            fig.append(r)
    elif axis == 'horizontal':
        xoffset = 0
        for i, r in enumerate(roots):
            size = newsizes[i]
            r.moveto(xoffset, 0, scale=scales[i])
            xoffset += size[0]
            fig.append(r)

    return fig


def extract_svg(display_object, dpi=300):
    """
    Removes the preamble of the svg files generated with nilearn
    """
    image_svg = svg2str(display_object, dpi)
    start_idx = image_svg.find('<svg ')
    end_idx = image_svg.rfind('</svg>')
    return image_svg[start_idx:end_idx]
