"""Arrival-ordered concurrent dispatch of agent tool calls.

Agent backends may deliver several tool calls at once. Read-only calls may overlap, but a call that can change
Harness or world state must observe every earlier call's effect and must not race a later read. The policy is
defined once here and the per-call verdict comes from the central tool registry contract
(``ToolSpec.parallel`` via ``ToolRegistry.parallel_safe``); there is no name list in the transport.

Ordering rule (unchanged from the original Codex ``RecordedClient`` dispatcher):

* a parallel-safe call starts after the most recent ordered call has finished and may overlap other parallel-safe
  calls;
* any other (ordered) call starts only after every parallel-safe call admitted before it and the previous ordered
  call have finished.

So execution order equals arrival order for every ordered call. Without a classifier every call is ordered, which
is the safe direction.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures


def is_parallel_call(parallel_safe, params):
    """Return True only when ``parallel_safe(tool, arguments)`` explicitly returns True for ``params``.

    ``parallel_safe`` is a callable ``(tool_name: str, arguments: dict) -> bool`` such as
    ``ToolRegistry.parallel_safe``, or ``None`` (every call ordered). ``params`` is the native tool-call payload
    with ``tool`` and ``arguments`` keys. A missing classifier, a non-object payload, a non-``True`` verdict or a
    classifier exception all yield False, so a malformed call can never stop the transport's reader.
    """
    if parallel_safe is None or not isinstance(params, dict):
        return False
    try:
        return parallel_safe(params.get("tool"), params.get("arguments")) is True
    except Exception:
        return False


def _run_after(deps, fn, args):
    if deps:
        wait_futures(deps)
    return fn(*args)


class OrderedBarrier:
    """Pool-agnostic ordering barrier between parallel-safe and ordered calls.

    The barrier never runs work itself: ``schedule`` submits a wrapper to the caller's executor that first waits for
    the call's dependencies (see the module docstring for the rule), then calls ``fn(*args)``. A dependency's
    exception or cancellation only ends the wait; it is never re-raised in a later call, so one failed call cannot
    deadlock or poison the chain. Dependencies are always submitted earlier, so a FIFO executor such as
    ``ThreadPoolExecutor`` cannot deadlock on them. Thread-safe; the arrival order is the order in which
    ``schedule`` acquires the internal lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active_parallel = []
        self._last_ordered = None

    def schedule(self, pool, parallel, fn, *args):
        """Submit ``fn(*args)`` to ``pool`` behind its dependencies and return the executor's ``Future``.

        ``pool`` is any executor with ``submit`` (e.g. ``ThreadPoolExecutor``). ``parallel`` is the call's
        classification: True for a parallel-safe call, anything else for an ordered call. The returned future
        carries ``fn``'s result or exception. Errors raised by ``pool.submit`` (e.g. after shutdown) propagate.
        """
        with self._lock:
            self._active_parallel = [future for future in self._active_parallel if not future.done()]
            ordered_dep = ([self._last_ordered]
                           if self._last_ordered is not None and not self._last_ordered.done() else [])
            if parallel is True:
                future = pool.submit(_run_after, ordered_dep, fn, args)
                self._active_parallel.append(future)
                return future
            deps = [*self._active_parallel, *ordered_dep]
            self._active_parallel = []
            future = pool.submit(_run_after, deps, fn, args)
            self._last_ordered = future
            return future


class ToolDispatcher:
    """Own a worker pool that runs native tool calls under the ``OrderedBarrier`` policy.

    ``handler(params) -> result`` executes one tool call (normally ``Harness.handle_tool`` behind the session's
    stale-connection checks); it runs on a worker thread, never on the caller's thread. ``parallel_safe`` is a
    ``(tool_name, arguments) -> bool`` classifier (``ToolRegistry.parallel_safe``) or ``None`` to order every call.
    ``max_workers`` bounds concurrently running calls (default 8, as the Codex transport); ``thread_name_prefix``
    names the worker threads for diagnostics.
    """

    def __init__(self, handler, parallel_safe, *, max_workers=8, thread_name_prefix="agent-tools"):
        self.handler = handler
        self.parallel_safe = parallel_safe
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self.barrier = OrderedBarrier()

    def submit(self, params):
        """Schedule ``handler(params)`` and return its ``Future``.

        ``params`` is the native tool-call payload (at least ``tool`` and ``arguments``); it is classified with
        ``is_parallel_call`` at submission, so submission order is the arrival order. Raises ``RuntimeError`` after
        ``shutdown``.
        """
        return self.barrier.schedule(self.pool, is_parallel_call(self.parallel_safe, params), self.handler, params)

    def shutdown(self, *, wait=True, cancel_futures=True):
        """Stop accepting calls; by default cancel queued calls and join the running ones.

        Join only after the robot is closed (session close ordering), so in-flight handlers can return.
        """
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)
