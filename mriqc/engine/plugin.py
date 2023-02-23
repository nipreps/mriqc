# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""A lightweight NiPype MultiProc execution plugin."""

# Import packages
import os
import sys
from copy import deepcopy
from time import sleep, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from traceback import format_exception
import gc


# Run node
def run_node(node, updatehash, taskid):
    """
    Execute node.run(), catch and log any errors and get a result.

    Parameters
    ----------
    node : nipype Node instance
        the node to run
    updatehash : boolean
        flag for updating hash
    taskid : int
        an identifier for this task
    Returns
    -------
    result : dictionary
        dictionary containing the node runtime results and stats

    """
    # Init variables
    result = dict(result=None, traceback=None, taskid=taskid)

    # Try and execute the node via node.run()
    try:
        result["result"] = node.run(updatehash=updatehash)
    except:  # noqa: E722, intendedly catch all here
        result["traceback"] = format_exception(*sys.exc_info())
        result["result"] = node.result

    # Return the result dictionary
    return result


class PluginBase:
    """Base class for plugins."""

    def __init__(self, plugin_args=None):
        """Initialize plugin."""
        if plugin_args is None:
            plugin_args = {}
        self.plugin_args = plugin_args
        self._config = None
        self._status_callback = plugin_args.get("status_callback")

    def run(self, graph, config, updatehash=False):
        """
        Instruct the plugin to execute the workflow graph.

        The core plugin member that should be implemented by
        all plugins.

        Parameters
        ----------
        graph :
            a networkx, flattened :abbr:`DAG (Directed Acyclic Graph)`
            to be executed
        config : :obj:`~nipype.config`
            a nipype.config object
        updatehash : :obj:`bool`
            whether cached nodes with stale hash should be just updated.

        """
        raise NotImplementedError


class DistributedPluginBase(PluginBase):
    """
    Execute workflow with a distribution engine.

    Combinations of ``proc_done`` and ``proc_pending``:
    +------------+---------------+--------------------------------+
    | proc_done  | proc_pending  | outcome                        |
    +============+===============+================================+
    | True       | False         | Process is finished            |
    +------------+---------------+--------------------------------+
    | True       | True          | Process is currently being run |
    +------------+---------------+--------------------------------+
    | False      | False         | Process is queued              |
    +------------+---------------+--------------------------------+
    | False      | True          | INVALID COMBINATION            |
    +------------+---------------+--------------------------------+

    Attributes
    ----------
    procs : :obj:`list`
        list (N) of underlying interface elements to be processed
    proc_done : :obj:`numpy.ndarray`
        a boolean numpy array (N,) signifying whether a process has been
        submitted for execution
    proc_pending : :obj:`numpy.ndarray`
        a boolean numpy array (N,) signifying whether a
        process is currently running.
    depidx : :obj:`numpy.matrix`
        a boolean matrix (NxN) storing the dependency structure across
        processes. Process dependencies are derived from each column.

    """

    def __init__(self, plugin_args=None):
        """Initialize runtime attributes to none."""
        super(DistributedPluginBase, self).__init__(plugin_args=plugin_args)
        self.procs = None
        self.depidx = None
        self.refidx = None
        self.mapnodes = None
        self.mapnodesubids = None
        self.proc_done = None
        self.proc_pending = None
        self.pending_tasks = []
        self.max_jobs = self.plugin_args.get("max_jobs", None)

    def _prerun_check(self, graph):
        """Stub method to validate/massage graph and nodes before running."""

    def _postrun_check(self):
        """Stub method to close any open resources."""

    def run(self, graph, config, updatehash=False):
        """Execute a pre-defined pipeline using distributed approaches."""
        import numpy as np

        self._config = config
        poll_sleep_secs = float(config["execution"]["poll_sleep_duration"])

        self._prerun_check(graph)
        # Generate appropriate structures for worker-manager model
        self._generate_dependency_list(graph)
        self.mapnodes = []
        self.mapnodesubids = {}
        # setup polling - TODO: change to threaded model
        notrun = []
        errors = []

        while not np.all(self.proc_done) or np.any(self.proc_pending):
            loop_start = time()
            toappend = []
            # trigger callbacks for any pending results
            while self.pending_tasks:
                taskid, jobid = self.pending_tasks.pop()
                try:
                    result = self._get_result(taskid)
                except Exception as exc:
                    notrun.append(self._clean_queue(jobid, graph))
                    errors.append(exc)
                else:
                    if result:
                        if result["traceback"]:
                            notrun.append(
                                self._clean_queue(jobid, graph, result=result)
                            )
                            errors.append("".join(result["traceback"]))
                        else:
                            self._task_finished_cb(jobid)
                            self._remove_node_dirs()
                        self._clear_task(taskid)
                    else:
                        assert self.proc_done[jobid] and self.proc_pending[jobid]
                        toappend.insert(0, (taskid, jobid))

            if toappend:
                self.pending_tasks.extend(toappend)

            num_jobs = len(self.pending_tasks)
            if self.max_jobs is None or num_jobs < self.max_jobs:
                self._send_procs_to_workers(updatehash=updatehash, graph=graph)

            sleep_til = loop_start + poll_sleep_secs
            sleep(max(0, sleep_til - time()))

        self._remove_node_dirs()

        # close any open resources
        self._postrun_check()

        if errors:
            # If one or more nodes failed, re-rise first of them
            error, cause = errors[0], None
            if isinstance(error, str):
                error = RuntimeError(error)

            if len(errors) > 1:
                error, cause = (
                    RuntimeError(f"{len(errors)} raised. Re-raising first."),
                    error,
                )

            raise error from cause

    def _get_result(self, taskid):
        raise NotImplementedError

    def _submit_job(self, node, updatehash=False):
        raise NotImplementedError

    def _report_crash(self, node, result=None):
        from nipype.pipeline.plugins.tools import report_crash

        tb = None
        if result is not None:
            node._result = result["result"]
            tb = result["traceback"]
            node._traceback = tb
        return report_crash(node, traceback=tb)

    def _clear_task(self, taskid):
        raise NotImplementedError

    def _clean_queue(self, jobid, graph, result=None):
        from mriqc import config

        if self._status_callback:
            self._status_callback(self.procs[jobid], "exception")
        if result is None:
            result = {
                "result": None,
                "traceback": "\n".join(format_exception(*sys.exc_info())),
            }

        crashfile = self._report_crash(self.procs[jobid], result=result)
        if config.nipype.stop_on_first_crash:
            raise RuntimeError("".join(result["traceback"]))
        if jobid in self.mapnodesubids:
            # remove current jobid
            self.proc_pending[jobid] = False
            self.proc_done[jobid] = True
            # remove parent mapnode
            jobid = self.mapnodesubids[jobid]
            self.proc_pending[jobid] = False
            self.proc_done[jobid] = True
        # remove dependencies from queue
        return self._remove_node_deps(jobid, crashfile, graph)

    def _send_procs_to_workers(self, updatehash=False, graph=None):
        """Submit tasks to workers when system resources are available."""

    def _submit_mapnode(self, jobid):
        import numpy as np
        import scipy.sparse as ssp

        if jobid in self.mapnodes:
            return True
        self.mapnodes.append(jobid)
        mapnodesubids = self.procs[jobid].get_subnodes()
        numnodes = len(mapnodesubids)
        for i in range(numnodes):
            self.mapnodesubids[self.depidx.shape[0] + i] = jobid
        self.procs.extend(mapnodesubids)
        self.depidx = ssp.vstack(
            (self.depidx, ssp.lil_matrix(np.zeros((numnodes, self.depidx.shape[1])))),
            "lil",
        )
        self.depidx = ssp.hstack(
            (self.depidx, ssp.lil_matrix(np.zeros((self.depidx.shape[0], numnodes)))),
            "lil",
        )
        self.depidx[-numnodes:, jobid] = 1
        self.proc_done = np.concatenate(
            (self.proc_done, np.zeros(numnodes, dtype=bool))
        )
        self.proc_pending = np.concatenate(
            (self.proc_pending, np.zeros(numnodes, dtype=bool))
        )
        return False

    def _local_hash_check(self, jobid, graph):
        from mriqc import config

        if not config.nipype.local_hash_check:
            return False

        try:
            cached, updated = self.procs[jobid].is_cached()
        except Exception:
            return False

        overwrite = self.procs[jobid].overwrite
        always_run = self.procs[jobid].interface.always_run

        if (
            cached
            and updated
            and (overwrite is False or overwrite is None and not always_run)
        ):
            try:
                self._task_finished_cb(jobid, cached=True)
                self._remove_node_dirs()
            except Exception:
                self._clean_queue(jobid, graph)
                self.proc_pending[jobid] = False
            return True
        return False

    def _task_finished_cb(self, jobid, cached=False):
        """
        Extract outputs and assign to inputs of dependent tasks.

        This is called when a job is completed.
        """
        if self._status_callback:
            self._status_callback(self.procs[jobid], "end")
        # Update job and worker queues
        self.proc_pending[jobid] = False
        # update the job dependency structure
        rowview = self.depidx.getrowview(jobid)
        rowview[rowview.nonzero()] = 0
        if jobid not in self.mapnodesubids:
            self.refidx[self.refidx[:, jobid].nonzero()[0], jobid] = 0

    def _generate_dependency_list(self, graph):
        """Generate a dependency list for a list of graphs."""
        import numpy as np
        import networkx as nx
        from nipype.pipeline.engine.utils import topological_sort

        self.procs, _ = topological_sort(graph)
        self.depidx = nx.to_scipy_sparse_matrix(
            graph, nodelist=self.procs, format="lil"
        )
        self.refidx = self.depidx.astype(int)
        self.proc_done = np.zeros(len(self.procs), dtype=bool)
        self.proc_pending = np.zeros(len(self.procs), dtype=bool)

    def _remove_node_deps(self, jobid, crashfile, graph):
        import networkx as nx

        try:
            dfs_preorder = nx.dfs_preorder
        except AttributeError:
            dfs_preorder = nx.dfs_preorder_nodes
        subnodes = [s for s in dfs_preorder(graph, self.procs[jobid])]
        for node in subnodes:
            idx = self.procs.index(node)
            self.proc_done[idx] = True
            self.proc_pending[idx] = False
        return dict(node=self.procs[jobid], dependents=subnodes, crashfile=crashfile)

    def _remove_node_dirs(self):
        """Remove directories whose outputs have already been used up."""
        import numpy as np
        from shutil import rmtree
        from mriqc import config

        if config.nipype.remove_node_directories:
            indices = np.nonzero((self.refidx.sum(axis=1) == 0).__array__())[0]
            for idx in indices:
                if idx in self.mapnodesubids:
                    continue
                if self.proc_done[idx] and (not self.proc_pending[idx]):
                    self.refidx[idx, idx] = -1
                    outdir = self.procs[idx].output_dir()
                    rmtree(outdir)


class MultiProcPlugin(DistributedPluginBase):
    """
    A lightweight re-implementation of NiPype's MultiProc plugin.

    Execute workflow with multiprocessing, not sending more jobs at once
    than the system can support.
    The plugin_args input to run can be used to control the multiprocessing
    execution and defining the maximum amount of memory and threads that
    should be used. When those parameters are not specified,
    the number of threads and memory of the system is used.
    System consuming nodes should be tagged::
      memory_consuming_node.mem_gb = 8
      thread_consuming_node.n_procs = 16

    The default number of threads and memory are set at node
    creation, and are 1 and 0.25GB respectively.
    Currently supported options are:
    - non_daemon: boolean flag to execute as non-daemon processes
    - n_procs: maximum number of threads to be executed in parallel
    - memory_gb: maximum memory (in GB) that can be used at once.
    - raise_insufficient: raise error if the requested resources for
        a node over the maximum `n_procs` and/or `memory_gb`
        (default is ``True``).
    - scheduler: sort jobs topologically (``'tsort'``, default value)
        or prioritize jobs by, first, memory consumption and, second,
        number of threads (``'mem_thread'`` option).
    - mp_context: name of multiprocessing context to use
    """

    def __init__(self, pool=None, plugin_args=None):
        """Initialize the plugin."""
        from mriqc import config

        super(MultiProcPlugin, self).__init__(plugin_args=plugin_args)
        self._taskresult = {}
        self._task_obj = {}
        self._taskid = 0

        # Cache current working directory and make sure we
        # change to it when workers are set up
        self._cwd = os.getcwd()

        # Read in options or set defaults.
        self.processors = self.plugin_args.get("n_procs", mp.cpu_count())
        self.memory_gb = self.plugin_args.get(
            "memory_gb",  # Allocate 90% of system memory
            config.environment.total_memory * 0.9,
        )
        self.raise_insufficient = self.plugin_args.get("raise_insufficient", False)

        # Instantiate different thread pools for non-daemon processes
        mp_context = mp.get_context(self.plugin_args.get("mp_context"))
        self.pool = pool or ProcessPoolExecutor(
            max_workers=self.processors,
            initializer=config._process_initializer,
            initargs=(config.execution.cwd, config.nipype.omp_nthreads),
            mp_context=mp_context,
        )

        self._stats = None

    def _async_callback(self, args):
        result = args.result()
        self._taskresult[result["taskid"]] = result

    def _get_result(self, taskid):
        return self._taskresult.get(taskid)

    def _clear_task(self, taskid):
        del self._task_obj[taskid]

    def _submit_job(self, node, updatehash=False):
        self._taskid += 1

        # Don't allow streaming outputs
        if getattr(node.interface, "terminal_output", "") == "stream":
            node.interface.terminal_output = "allatonce"

        result_future = self.pool.submit(run_node, node, updatehash, self._taskid)
        result_future.add_done_callback(self._async_callback)
        self._task_obj[self._taskid] = result_future
        return self._taskid

    def _prerun_check(self, graph):
        """Check if any node exceeds the available resources."""
        import numpy as np

        tasks_mem_gb = []
        tasks_num_th = []
        for node in graph.nodes():
            tasks_mem_gb.append(node.mem_gb)
            tasks_num_th.append(node.n_procs)

        if self.raise_insufficient and (
            np.any(np.array(tasks_mem_gb) > self.memory_gb)
            or np.any(np.array(tasks_num_th) > self.processors)
        ):
            raise RuntimeError("Insufficient resources available for job")

    def _postrun_check(self):
        self.pool.shutdown()

    def _check_resources(self, running_tasks):
        """Make sure there are resources available."""
        free_memory_gb = self.memory_gb
        free_processors = self.processors
        for _, jobid in running_tasks:
            free_memory_gb -= min(self.procs[jobid].mem_gb, free_memory_gb)
            free_processors -= min(self.procs[jobid].n_procs, free_processors)

        return free_memory_gb, free_processors

    def _send_procs_to_workers(self, updatehash=False, graph=None):
        """Submit tasks to workers when system resources are available."""
        import numpy as np

        # Check to see if a job is available (jobs with all dependencies run)
        # See https://github.com/nipy/nipype/pull/2200#discussion_r141605722
        # See also https://github.com/nipy/nipype/issues/2372
        jobids = np.flatnonzero(
            ~self.proc_done & (self.depidx.sum(axis=0) == 0).__array__()
        )

        # Check available resources by summing all threads and memory used
        free_memory_gb, free_processors = self._check_resources(self.pending_tasks)

        stats = (
            len(self.pending_tasks),
            len(jobids),
            free_memory_gb,
            self.memory_gb,
            free_processors,
            self.processors,
        )
        if self._stats != stats:
            self._stats = stats

        if free_memory_gb < 0.01 or free_processors == 0:
            return

        if len(jobids) + len(self.pending_tasks) == 0:
            return

        jobids = self._sort_jobs(jobids, scheduler=self.plugin_args.get("scheduler"))

        # Run garbage collector before potentially submitting jobs
        gc.collect()

        # Submit jobs
        for jobid in jobids:
            # First expand mapnodes
            if self.procs[jobid].__class__.__name__ == "MapNode":
                try:
                    num_subnodes = self.procs[jobid].num_subnodes()
                except Exception:
                    traceback = format_exception(*sys.exc_info())
                    self._clean_queue(
                        jobid, graph, result={"result": None, "traceback": traceback}
                    )
                    self.proc_pending[jobid] = False
                    continue
                if num_subnodes > 1:
                    submit = self._submit_mapnode(jobid)
                    if not submit:
                        continue

            # Check requirements of this job
            next_job_gb = min(self.procs[jobid].mem_gb, self.memory_gb)
            next_job_th = min(self.procs[jobid].n_procs, self.processors)

            # If node does not fit, skip at this moment
            if next_job_th > free_processors or next_job_gb > free_memory_gb:
                continue

            free_memory_gb -= next_job_gb
            free_processors -= next_job_th
            # change job status in appropriate queues
            self.proc_done[jobid] = True
            self.proc_pending[jobid] = True

            # If cached and up-to-date just retrieve it, don't run
            if self._local_hash_check(jobid, graph):
                continue

            # updatehash and run_without_submitting are also run locally
            if updatehash or self.procs[jobid].run_without_submitting:
                try:
                    self.procs[jobid].run(updatehash=updatehash)
                except Exception:
                    traceback = format_exception(*sys.exc_info())
                    self._clean_queue(
                        jobid, graph, result={"result": None, "traceback": traceback}
                    )

                # Release resources
                self._task_finished_cb(jobid)
                self._remove_node_dirs()
                free_memory_gb += next_job_gb
                free_processors += next_job_th
                # Display stats next loop
                self._stats = None

                # Clean up any debris from running node in main process
                gc.collect()
                continue

            # Task should be submitted to workers
            # Send job to task manager and add to pending tasks
            if self._status_callback:
                self._status_callback(self.procs[jobid], "start")
            tid = self._submit_job(deepcopy(self.procs[jobid]), updatehash=updatehash)
            if tid is None:
                self.proc_done[jobid] = False
                self.proc_pending[jobid] = False
            else:
                self.pending_tasks.insert(0, (tid, jobid))
            # Display stats next loop
            self._stats = None

    def _sort_jobs(self, jobids, scheduler="tsort"):
        if scheduler == "mem_thread":
            return sorted(
                jobids,
                key=lambda item: (self.procs[item].mem_gb, self.procs[item].n_procs),
            )
        return jobids
