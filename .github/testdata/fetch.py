# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Assemble a test dataset for MRIQC's integration tests from pinned public sources.

Each manifest lists files on public S3 buckets, pinned by object version and MD5.
The script downloads and checks them, keeps only some DWI volumes when the manifest
says so, and writes a minimal BIDS dataset.

Usage::

    uv run --with nibabel --with numpy python fetch.py MANIFEST OUT_DIR
    uv run --with nibabel --with numpy python fetch.py --select MANIFEST PREFIX N

The second form prints which volumes of a DWI run (``PREFIX``, a path in the
manifest without extension) to keep: every b=0 volume, and ``N`` directions per
shell spread over the sphere. Paste its output into the manifest's ``subsets``.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import nibabel as nb
import numpy as np

B0_THRESHOLD = 50


def download(entry, dest, retries=5):
    """Download one pinned file to ``dest`` and check its MD5."""
    url = f'{entry["url"]}?versionId={entry["version_id"]}'
    for attempt in range(1, retries + 1):
        try:
            md5 = hashlib.md5(usedforsecurity=False)
            with urllib.request.urlopen(url, timeout=60) as resp, open(dest, 'wb') as out:
                while chunk := resp.read(1 << 20):
                    md5.update(chunk)
                    out.write(chunk)
        except OSError as exc:
            if attempt == retries:
                raise
            print(f'  retrying {url} ({exc})', file=sys.stderr)
            time.sleep(2**attempt)
            continue
        if md5.hexdigest() != entry['md5']:
            raise RuntimeError(f'MD5 mismatch for {url}: {md5.hexdigest()} != {entry["md5"]}')
        return dest


def select_volumes(bvals, bvecs, per_shell):
    """Keep every b=0, and ``per_shell`` directions per shell by farthest-point sampling.

    Distances are antipodally symmetric (a direction and its opposite are the same).
    The first direction of each shell seeds the sampling, so the result is deterministic.
    """
    keep = list(np.flatnonzero(bvals < B0_THRESHOLD))
    shells = np.round(bvals, -2)
    for shell in sorted(set(shells[bvals >= B0_THRESHOLD])):
        index = np.flatnonzero((bvals >= B0_THRESHOLD) & (shells == shell))
        vecs = bvecs[:, index].T
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        chosen = [0]
        distance = 1 - np.abs(vecs @ vecs[0])
        while len(chosen) < min(per_shell, len(index)):
            chosen.append(int(np.argmax(distance)))
            distance = np.minimum(distance, 1 - np.abs(vecs @ vecs[chosen[-1]]))
        keep += list(index[chosen])
    return sorted(int(i) for i in keep)


def write_subset(prefix, sources, volumes, out_dir):
    """Write the selected volumes of a DWI run, with its b-values and b-vectors."""
    img = nb.load(sources[f'{prefix}.nii.gz'])
    data = np.asanyarray(img.dataobj.get_unscaled())[..., volumes]
    subset = img.__class__(data, img.affine, img.header)
    subset.header.set_slope_inter(*img.header.get_slope_inter())
    subset.to_filename(out_dir / f'{prefix}.nii.gz')

    bvals = np.loadtxt(sources[f'{prefix}.bval'], ndmin=1)
    bvecs = np.loadtxt(sources[f'{prefix}.bvec'], ndmin=2)
    (out_dir / f'{prefix}.bval').write_text(' '.join(f'{v:g}' for v in bvals[volumes]) + '\n')
    (out_dir / f'{prefix}.bvec').write_text(
        '\n'.join(' '.join(f'{v:.6f}' for v in row) for row in bvecs[:, volumes]) + '\n'
    )


def readme(manifest):
    """Describe the dataset, where each file comes from, and its licence."""
    lines = [f'# {manifest["dataset_description"]["Name"]}', '', manifest['description'], '']
    lines += ['## Sources', '']
    for source in manifest['sources']:
        lines.append(f'- **{source["name"]}** ({source["license"]}): {source["url"]}')
        if source.get('attribution'):
            lines.append(f'  {source["attribution"]}')
    lines += [
        '',
        'Every file is pinned by its S3 object version and MD5 in `.github/testdata/` of',
        'https://github.com/nipreps/mriqc, where `fetch.py` rebuilds this dataset.',
    ]
    if manifest.get('subsets'):
        lines += [
            '',
            '## Volume selection',
            '',
            'DWI runs keep every b=0 volume and some directions per shell, chosen by',
            'farthest-point sampling on the sphere (`fetch.py --select`).',
            'The kept volumes (0-based) are listed in the manifest.',
        ]
    return '\n'.join(lines) + '\n'


def assemble(manifest_file, out_dir):
    """Build the dataset in a temporary folder, and move it into place when complete."""
    manifest = json.loads(Path(manifest_file).read_text())
    out_dir = Path(out_dir)
    if out_dir.exists():
        sys.exit(f'{out_dir} exists already; remove it first.')
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        tmp = Path(tmp)
        staging, downloads = tmp / 'dataset', tmp / 'downloads'
        sources = {}
        for entry in manifest['files']:
            print(f'Fetching {entry["path"]}')
            target = downloads / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            sources[entry['path']] = download(entry, target)

        subsets = manifest.get('subsets', {})
        subset_files = {f'{p}{ext}' for p in subsets for ext in ('.nii.gz', '.bval', '.bvec')}
        for path, src in sources.items():
            if path in subset_files or path == 'participants.tsv':
                continue
            (staging / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, staging / path)
        for prefix, volumes in subsets.items():
            (staging / prefix).parent.mkdir(parents=True, exist_ok=True)
            write_subset(prefix, sources, volumes, staging)

        subjects = sorted(p.name for p in staging.glob('sub-*') if p.is_dir())
        if 'participants.tsv' in sources:
            # Keep the header and the rows of the subjects in this dataset
            rows = sources['participants.tsv'].read_text().splitlines()
            kept = [rows[0]] + sorted(r for r in rows[1:] if r.split('\t')[0] in subjects)
        else:
            kept = ['participant_id'] + subjects
        (staging / 'participants.tsv').write_text('\n'.join(kept) + '\n')

        description = manifest['dataset_description']
        (staging / 'dataset_description.json').write_text(json.dumps(description, indent=2) + '\n')
        (staging / 'README.md').write_text(readme(manifest))

        staging.rename(out_dir)
    print(f'Wrote {out_dir} ({len(subjects)} subjects)')


def print_selection(manifest_file, prefix, per_shell):
    """Download a DWI run's b-values and b-vectors, and print the volumes to keep."""
    entries = {e['path']: e for e in json.loads(Path(manifest_file).read_text())['files']}
    with tempfile.TemporaryDirectory() as tmp:
        bvals = np.loadtxt(download(entries[f'{prefix}.bval'], Path(tmp) / 'bval'), ndmin=1)
        bvecs = np.loadtxt(download(entries[f'{prefix}.bvec'], Path(tmp) / 'bvec'), ndmin=2)
    print(json.dumps(select_volumes(bvals, bvecs, per_shell)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('manifest', help='JSON manifest of the dataset')
    parser.add_argument('out_dir', nargs='?', help='folder to write the dataset to')
    parser.add_argument(
        '--select',
        nargs=2,
        metavar=('PREFIX', 'N'),
        help='print the volumes of a DWI run to keep, with N directions per shell',
    )
    opts = parser.parse_args()
    if opts.select:
        print_selection(opts.manifest, opts.select[0], int(opts.select[1]))
    elif opts.out_dir:
        assemble(opts.manifest, opts.out_dir)
    else:
        parser.error('OUT_DIR is required unless --select is given')


if __name__ == '__main__':
    main()
