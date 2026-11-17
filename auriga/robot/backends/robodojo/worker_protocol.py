"""JSONL worker dispatch and cancellation; the caller owns the native adapter.

All adapter operations remain on the caller thread; only cancellation reads run
on the input thread. Owner EOF is a process-exit boundary, never an action retry.
"""
import json
import os
import queue
import sys
import threading
import traceback

WIRE_PREFIX = "AURIGA_JSON "


def serve_worker(adapter_factory, arguments, emit):
    requests, cancellations = queue.Queue(), {}
    lock = threading.Lock()

    def read_input():
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["op"] == "cancel":
                    with lock:
                        event = cancellations.get(request.get("target"))
                        if event:
                            event.set()
                    emit({"id": request["id"], "result": {"stop_requested": event is not None, "confirmed": False}})
                else:
                    if request["op"] == "execute":
                        with lock:
                            cancellations[request["id"]] = threading.Event()
                    requests.put(request)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        # Input owner is gone: prevent any further execution, then close on the Isaac thread.
        with lock:
            for event in cancellations.values():
                event.set()
        # This simulator process has one IPC owner. Losing it must halt even a native GPU planner.
        os._exit(0)

    threading.Thread(target=read_input, daemon=True).start()
    try:
        adapter = adapter_factory(arguments)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        while not requests.empty():
            request = requests.get_nowait()
            emit({"id": request["id"], "error": f"bootstrap_failed: {type(error).__name__}: {error}"})
        sys.stderr.flush()
        os._exit(1)
    while True:
        request = requests.get()
        request_id, operation = request["id"], request["op"]
        try:
            if operation == "ready":
                result = adapter.capabilities()
            elif operation == "observe":
                result = adapter.observe()
            elif operation == "policy_observation":
                result = adapter.policy_observation()
            elif operation == "preview_motion":
                result = adapter.preview_motion(request["payload"])
            elif operation == "preview_batch":
                result = adapter.preview_batch(request["payload"])
            elif operation == "reset":
                result = adapter.reset(**request["payload"])
            elif operation == "evaluate":
                result = adapter.evaluate(**request["payload"])
            elif operation == "observer_snapshot":
                result = adapter.observer_snapshot()
            elif operation == "record":
                result = adapter.record(**request["payload"])
            elif operation == "save_video":
                result = adapter.save_video()
            elif operation == "fast_forward_slice":
                result = adapter.fast_forward_slice(request["payload"])
            elif operation == "execute":
                with lock:
                    cancellation = cancellations.setdefault(request_id, threading.Event())
                result = adapter.execute(request["payload"], lambda update: emit({"id": request_id, "progress": update}), cancellation)
                with lock:
                    cancellations.pop(request_id, None)
            elif operation == "close":
                adapter.close(lambda update: emit({"id": request_id, "progress": update}))
                emit({"id": request_id, "result": {"closed": True}})
                # Full native cleanup has returned and emit has flushed the ACK.
                # Python finalizers can still reference the unloaded Kit framework.
                sys.stderr.flush()
                os._exit(0)
            else:
                raise ValueError("unsupported operation")
            emit({"id": request_id, "result": result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({"id": request_id, "error": f"{type(error).__name__}: {error}"})
