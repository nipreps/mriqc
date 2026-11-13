#!/usr/bin/env python3
"""Summarize where an MRIQC run spends its time, node by node.

The source is either a work directory or a log written with ``-vv``:

* In a work directory, each node's ``_report/report.rst`` gives its duration, and the
  file's modification time gives its end, because *nipype* rewrites the report when the
  node finishes.
  Profile a fresh work directory: nodes reused from a previous run keep their old reports.
* In a log, each ``[Node] Finished "<name>", elapsed time <seconds>s.`` line gives the
  end and the duration of a node, but only its short name.

The summary lists the total, median and share of time per node, the wall time (from
the first node's start to the last node's end), the linear time (the sum of all node
durations, as if they ran one at a time), the average number of nodes running at once
(their ratio), and the critical chain: tracing back from the node that ended last, the
node that ended last before each one started.
Gaps in that chain are time spent outside node execution (workflow set-up, hashing,
waiting for free processors).

Only the standard library is used, so the script runs on any Python 3.11 or newer::

    python3 .maint/profile_nodes.py <work_dir or log> [--top 20] [--markdown] [--tsv out.tsv]

"""

import argparse
import bisect
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

HASH = re.compile(r'^[0-9a-f]{32,40}$')
DURATION = re.compile(r'^\* duration : ([\d.]+)', re.MULTILINE)
FINISHED = re.compile(r'\[Node\] Finished "([^"]+)", elapsed time ([\d.]+)s')
TIMESTAMP = re.compile(r'(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:[.,](\d+))?')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


@dataclass(frozen=True)
class Record:
    name: str
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start


def node_key(report_dir, root):
    """
    Name a node after its folder, without hashes and *MapNode* iterations.

    Examples
    --------
    >>> node_key(Path('/w/mriqc_wf/anatMRIQC/synthstrip_wf/' + 'a' * 40 + '/pre_n4'), Path('/w'))
    'anatMRIQC.synthstrip_wf.pre_n4'
    >>> node_key(Path('/w/mriqc_wf/anatMRIQC/tpms_std2t1w/mapflow/_tpms_std2t1w2'), Path('/w'))
    'anatMRIQC.tpms_std2t1w'

    """
    parts = report_dir.relative_to(root).parts
    if 'mapflow' in parts:
        parts = parts[: parts.index('mapflow')]
    parts = [part for part in parts if not HASH.match(part)]
    if parts and parts[0] == 'mriqc_wf':
        parts = parts[1:]
    return '.'.join(parts)


def log_key(name):
    """
    Name a node from a log line, merging *MapNode* iterations.

    Examples
    --------
    >>> log_key('_apply_hmc0')
    'apply_hmc'
    >>> log_key('EPI2MNI')
    'EPI2MNI'

    """
    match = re.match(r'^_(.+?)\d+$', name)
    return match.group(1) if match else name


def parse_timestamp(line):
    """
    Read the first timestamp of a log line, in seconds since the epoch.

    Log timestamps carry no time zone; they are read as UTC, which the container and
    GitHub Actions use.

    Examples
    --------
    >>> t0 = parse_timestamp('2026-11-09 09:53:40 | INFO | nipype.workflow')
    >>> round(parse_timestamp('2026-11-09T09:53:40.5124592Z 2026-11-09 09:53:40 | INFO') - t0, 3)
    0.512

    """
    match = TIMESTAMP.search(line)
    if not match:
        return None
    day, time, fraction = match.groups()
    value = datetime.fromisoformat(f'{day}T{time}').replace(tzinfo=UTC).timestamp()
    if fraction:
        value += float(f'0.{fraction[:6]}')
    return value


def from_work_dir(root):
    records = []
    for report in root.rglob('_report/report.rst'):
        found = DURATION.search(report.read_text(errors='ignore'))
        if not found:
            continue  # the node did not finish
        end = report.stat().st_mtime
        name = node_key(report.parent.parent, root)
        records.append(Record(name, end - float(found.group(1)), end))
    return records


def from_log(path):
    records = []
    for line in path.read_text(errors='ignore').splitlines():
        line = ANSI.sub('', line)
        found = FINISHED.search(line)
        end = parse_timestamp(line) if found else None
        if end is not None:
            records.append(Record(log_key(found.group(1)), end - float(found.group(2)), end))
    return records


def critical_chain(records, slack=1.0):
    """
    Trace back from the last node to end, through the last node to end before each start.

    ``slack`` (seconds) absorbs the rounding of log timestamps.

    Examples
    --------
    >>> recs = [Record('a', 0, 10), Record('b', 0, 4), Record('c', 10.5, 20), Record('d', 5, 12)]
    >>> [r.name for r in critical_chain(recs)]
    ['a', 'c']

    """
    by_end = sorted(records, key=lambda r: r.end)
    ends = [r.end for r in by_end]
    chain = [by_end[-1]]
    while True:
        current = chain[-1]
        i = bisect.bisect_right(ends, current.start + slack) - 1
        while i >= 0 and (by_end[i] is current or by_end[i].start >= current.start):
            i -= 1
        if i < 0:
            break
        chain.append(by_end[i])
    return chain[::-1]


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record.name, []).append(record.duration)
    linear = sum(sum(durations) for durations in groups.values())
    rows = [
        (
            name,
            len(durations),
            sum(durations),
            statistics.median(durations),
            sum(durations) / linear,
        )
        for name, durations in groups.items()
    ]
    return sorted(rows, key=lambda row: -row[2]), linear


def fmt(seconds):
    return f'{seconds / 60:.1f} min' if seconds >= 600 else f'{seconds:.1f} s'


def report(records, top, markdown):
    rows, linear = summarize(records)
    wall = max(r.end for r in records) - min(r.start for r in records)
    chain = critical_chain(records)
    chain_busy = sum(r.duration for r in chain)
    lines = []
    head = (
        f'{len(records)} node runs ({len(rows)} nodes): wall time {fmt(wall)}, '
        f'linear time {fmt(linear)}, on average {linear / wall:.1f} nodes running at once.'
    )
    chain_line = (
        f'Critical chain: {len(chain)} node runs, {fmt(chain_busy)} running and '
        f'{fmt(chain[-1].end - chain[0].start - chain_busy)} in gaps.'
    )
    header = ('Node', 'Runs', 'Total', 'Median', 'Share')
    table = [
        (name, str(runs), fmt(total), fmt(median), f'{share:.1%}')
        for name, runs, total, median, share in rows[:top]
    ]
    # Chain entries that last at least a second, or follow a gap of at least a second
    steps, previous_end = [], chain[0].start
    for record in chain:
        gap = max(0.0, record.start - previous_end)
        if record.duration >= 1.0 or gap >= 1.0:
            steps.append((fmt(gap), record.name, fmt(record.duration)))
        previous_end = record.end
    omitted = f'{len(chain) - len(steps)} shorter node runs not listed.'
    if markdown:
        lines += ['### Node times', '', head, '']
        lines += ['| ' + ' | '.join(header) + ' |', '|---|--:|--:|--:|--:|']
        lines += [f'| `{row[0]}` | ' + ' | '.join(row[1:]) + ' |' for row in table]
        lines += ['', '<details><summary>' + chain_line + '</summary>', '']
        lines += ['| Gap before | Node | Time |', '|--:|---|--:|']
        lines += [f'| {gap} | `{name}` | {time} |' for gap, name, time in steps]
        lines += ['', omitted, '', '</details>']
    else:
        widths = [max(len(row[i]) for row in [header, *table]) for i in range(5)]
        lines += [head, '']
        for row in [header, *table]:
            cells = [row[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(row[1:], widths[1:])]
            lines.append('  '.join(cells))
        lines += ['', chain_line, '  gap before      time  node']
        lines += [f'  {gap:>10}  {time:>8}  {name}' for gap, name, time in steps]
        lines += [f'  {omitted}']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', type=Path, help='a work directory, or a log written with -vv')
    parser.add_argument('--top', type=int, default=20, help='nodes to list (default: 20)')
    parser.add_argument('--markdown', action='store_true', help='print GitHub Markdown')
    parser.add_argument('--tsv', type=Path, help='also write one row per node run to this file')
    args = parser.parse_args(argv)

    records = from_work_dir(args.source) if args.source.is_dir() else from_log(args.source)
    if not records:
        print(f'No finished nodes found in {args.source}', file=sys.stderr)
        return 1

    print(report(records, args.top, args.markdown))
    if args.tsv:
        utc = UTC
        rows = ['node\tstart\tend\tduration']
        rows += [
            f'{r.name}\t{datetime.fromtimestamp(r.start, utc).isoformat()}\t'
            f'{datetime.fromtimestamp(r.end, utc).isoformat()}\t{r.duration:.3f}'
            for r in sorted(records, key=lambda r: r.start)
        ]
        args.tsv.write_text('\n'.join(rows) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
