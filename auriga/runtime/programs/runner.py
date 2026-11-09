"""Invocation-owned Python jobs with resumable main-actor action review."""
import base64
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import stat
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Callable

from ...core.contracts import canonical, digest, utc_now
from ...core.observation_policy import LEGACY_ACTOR_PROFILE, reject_privileged, allows_depth
from ...core.storage.store import Store
from ...cortex.session.system_runtime import runtime_library_paths, stdlib_runtime_entries, system_library_path

TERMINAL = {'succeeded','failed','cancelled','unknown'}
ACTION_RESERVE_S = 5.0


def program_runtime(root, *, sandbox_enabled=True):
    """Expose Python's runtime and selected computation packages, never its prefix.

    The private directory supplies import directory listings. Its links target
    individually allowed runtime entries; no host site-packages or .pth is read.
    Callers serialize creation and keep this directory outside actor write roots.
    """
    root = Path(root)
    library = root / 'lib' / ('python' + sysconfig.get_python_version())
    packages = library / 'site-packages'
    packages.mkdir(parents=True, exist_ok=True)
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    read_paths = [str(root), str(Path(sys.executable).resolve())]
    runtime_objects = {Path(sys.executable).resolve()}
    if sandbox_enabled:
        seccomp = system_library_path('libseccomp.so.2')
        runtime_objects.add(seccomp)
        read_paths.append(str(seccomp))

    def expose(source, destination, boundary):
        if source.is_symlink():
            raise ValueError('program_runtime_external_symlink')
        source = source.resolve()
        if not source.is_relative_to(boundary):
            raise ValueError('program_runtime_external_symlink')
        if source.is_dir():
            for item in source.rglob('*'):
                if item.is_symlink() and not item.resolve().is_relative_to(source):
                    raise ValueError('program_runtime_external_symlink')
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError('program_runtime_link_conflict')
        elif destination.exists():
            raise ValueError('program_runtime_link_conflict')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source, target_is_directory=source.is_dir())
        read_paths.append(str(source))
        if source.is_file():
            runtime_objects.add(source)
        elif source.is_dir():
            runtime_objects.update(path for path in source.rglob('*.so') if path.is_file())

    module_names = set(sys.stdlib_module_names) | {sysconfig._get_sysconfigdata_name()}
    for item in stdlib_runtime_entries(stdlib, module_names):
        expose(item, library / item.relative_to(stdlib), stdlib)
    for name, libraries in (('numpy', ('numpy.libs',)), ('scipy', ('scipy.libs',)),
                            ('PIL', ('pillow.libs',)),
                            ('cv2', ('opencv_python_headless.libs', 'opencv_python.libs'))):
        # A venv may inherit its base interpreter's site-packages. sysconfig's
        # purelib describes the venv destination, not where imports are found.
        spec = importlib.util.find_spec(name)
        if spec is None:
            continue
        if not spec.origin or not spec.submodule_search_locations:
            raise ValueError('program_runtime_requires_regular_package')
        source = Path(spec.origin).parent
        boundary = source.parent.resolve()
        expose(source, packages / name, boundary)
        for library in libraries:
            library_source = source.parent / library
            if library_source.exists():
                expose(library_source, packages / library, boundary)
    if sandbox_enabled:
        read_paths.extend(str(path) for path in runtime_library_paths(runtime_objects))
    return root, packages, sorted(set(read_paths))


@dataclass(frozen=True)
class ProgramServices:
    """Server-bound broker operations; no robot, registry, or runtime state owner."""
    store: Store
    validate_tool: Callable[[str], None]
    validate_arguments: Callable[[str, dict], None]
    invoke: Callable[[str, dict, dict], dict]
    effect: Callable[[str, dict], str]
    observation: Callable[[], dict]
    action_active: Callable[[str], bool]
    cancel_action: Callable[[str], bool]
    sandbox_enabled: bool = True
    workspace_modules: tuple = ()


def program_remaining(entry):
    """Remaining work time, excluding this job's own model/review waits."""
    lock = entry.setdefault('budget_lock', threading.RLock())
    with lock:
        started = entry.get('model_wait_started_at')
        if started is None:
            started = entry.get('review_wait_started_at')
        return entry['deadline'] - (time.monotonic() if started is None else started)


@contextmanager
def model_wait(entry):
    """Pause only this program's work clock around its counted Sol request.

    This does not change CPU/resource limits or action timeouts. Explicit
    cancellation remains active, and unrelated programs/actor calls earn no
    additional work time.
    """
    lock = entry.setdefault('budget_lock', threading.RLock())
    with lock:
        started = time.monotonic()
        if entry['cancel'].is_set() or started >= entry['deadline']:
            raise ValueError('program_channel_expired')
        if entry.get('model_wait_started_at') is not None or entry.get('review_wait_started_at') is not None:
            raise ValueError('program_model_wait_already_active')
        entry['model_wait_started_at'] = started
    try:
        yield
    finally:
        with lock:
            elapsed = max(0.0, time.monotonic() - started)
            entry['deadline'] += elapsed
            entry['model_wait_s'] = entry.get('model_wait_s', 0.0) + elapsed
            entry['model_wait_started_at'] = None


@contextmanager
def review_wait(entry):
    """Freeze only this job's work budget while its main actor reviews a plan."""
    lock = entry.setdefault('budget_lock', threading.RLock())
    with lock:
        started = time.monotonic()
        if entry['cancel'].is_set() or started >= entry['deadline']:
            raise ValueError('program_channel_expired')
        if entry.get('review_wait_started_at') is not None or entry.get('model_wait_started_at') is not None:
            raise ValueError('program_wait_already_active')
        entry['review_wait_started_at'] = started
    try:
        yield
    finally:
        with lock:
            elapsed = max(0.0, time.monotonic() - started)
            entry['deadline'] += elapsed
            entry['review_wait_s'] = entry.get('review_wait_s', 0.0) + elapsed
            entry['review_wait_started_at'] = None

class ProgramRunner:
    def __init__(self, services: ProgramServices):
        if not isinstance(services, ProgramServices):
            raise TypeError('ProgramRunner requires ProgramServices')
        self.services, self.store = services, services.store
        private = self.store.root / 'private'
        if private.is_symlink():
            raise ValueError('program_private_directory_must_be_real')
        private.mkdir(exist_ok=True, mode=0o700)
        self.root = private / 'program-workspaces'
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs = private / 'program-logs'
        self.logs.mkdir(parents=True, exist_ok=True)
        for path in (self.root.parent, self.root, self.logs):
            if path.is_symlink() or not path.is_dir():
                raise ValueError('program_private_directory_must_be_real')
            path.chmod(0o700)
        self.entries = {}
        self.last_activity_at = time.monotonic()
        self.lock = threading.RLock()
        self.runtime_lock = threading.Lock()
        self.runtime = None
        for job in self.store.states('program'):
            if job['status'] not in TERMINAL:
                job.update(status='unknown', stop_confirmed=False,
                           reason='gateway_restarted_without_program_acknowledgement')
                self.store.set_state('program',job['job_id'],job)

    def snapshot(self):
        """Expose lifecycle facts without lending out mutable entries or locks."""
        with self.lock:
            entries = list(self.entries.values())
        active, pending = [], []
        for entry in entries:
            with entry['condition']:
                active.append(entry['job_id'])
                job = entry['job']
                review = job.get('pending_review') or {}
                process = entry.get('process')
                if (job.get('status') == 'awaiting_review' and entry.get('process_paused')
                        and not entry['cancel'].is_set() and process is not None and process.poll() is None
                        and review.get('decision') == 'pending' and review.get('review_id')):
                    pending.append(dict(job_id=entry['job_id'], review_id=review['review_id'],
                        kind=review.get('kind'), **({'plan_id': review['plan_id']} if review.get('plan_id') else {})))
        # Unknown ACKs outlive their supervisor threads. Never infer a confirmed
        # stop from an empty entries map or erase old unresolved process ownership.
        # A job is published before its thread/entry starts. That startup window
        # is active ownership too, never permission for observer evaluation.
        active.extend(job['job_id'] for job in self.store.states('program')
                      if job['status'] not in TERMINAL and job['job_id'] not in active)
        return dict(active_jobs=active, pending_reviews=pending, unresolved_jobs=self.unresolved_jobs())

    def unresolved_jobs(self):
        return [job['job_id'] for job in self.store.states('program')
                if job.get('status') == 'unknown' or job.get('stop_confirmed') is False]

    def continuation_state(self):
        state = self.snapshot()
        blocked = bool(state['unresolved_jobs'] or len(state['active_jobs']) != len(state['pending_reviews']))
        return dict(blocked=blocked, pending_reviews=[] if blocked else state['pending_reviews'])

    def _active_entry(self, identifier):
        with self.lock:
            entry = self.entries.get(identifier)
        if entry is None or entry['cancel'].is_set() or program_remaining(entry) <= 0:
            raise ValueError('program_channel_expired')
        return entry

    def admit_action(self, identifier, timeout_s):
        if program_remaining(self._active_entry(identifier)) < timeout_s + ACTION_RESERVE_S:
            raise ValueError('program_time_insufficient_for_action')

    @staticmethod
    def _action_timeout(entry, tool, arguments):
        from ...robot.motion.arx_actions import DEFAULT_ACTION_TIMEOUT_S, DEFAULT_WAIT_TIMEOUT_S
        if 'timeout_s' in arguments:
            entry['_explicit_timeout'] = arguments['timeout_s']
            return arguments
        entry.pop('_explicit_timeout', None)
        if tool == 'robot_action' and arguments.get('mode', 'preview') in {'preview', 'direct', 'execute_if_clean'}:
            default = DEFAULT_ACTION_TIMEOUT_S
        elif tool == 'robot_wait':
            default = DEFAULT_WAIT_TIMEOUT_S
        else:
            return arguments
        # Leave dispatch headroom; expensive planning can still fail final admission.
        margin = 6.0 if program_remaining(entry) >= 20.0 else 1.0
        available = program_remaining(entry) - ACTION_RESERVE_S - margin
        if available < .1:
            if (
                tool == 'robot_action'
                and arguments.get('mode', 'preview') == 'preview'
                and not arguments.get('auto_execute_if_clean')
            ):
                return arguments  # Read-only planning need not reserve execution time.
            raise ValueError('program_time_insufficient_for_action')
        return {**arguments, 'timeout_s': min(default, available)}

    def review_controls(self, identifier):
        entry = self._active_entry(identifier)
        return dict(cancel_event=entry['cancel'], wait_scope=lambda: model_wait(entry))

    def cancel_all(self):
        with self.lock:
            identifiers = list(self.entries)
        for identifier in identifiers:
            self.cancel(identifier)

    def status(self, identifier):
        value = self.store.read_state('program',identifier)
        if not value: raise ValueError('unknown_program_job')
        result=dict(value)
        for name in ("stdout","stderr"):
            if value.get(name+"_ref"):
                info,content=self.store.resolve_artifact(value[name+"_ref"],allowed_scopes=("actor",))
                result[name+"_text"]=content[:8192].decode(errors="replace")
                result[name+"_truncated"]=len(content)>8192
        return result

    def review_content(self, identifier):
        """Original preview content for the main actor; image blocks stay images."""
        job = self.store.read_state('program', identifier)
        reference = (job or {}).get('pending_review', {}).get('preview_ref')
        if not reference:
            return []
        info, raw = self.store.resolve_artifact(reference, allowed_scopes=('actor',))
        if info.get('kind') != 'program_review_preview':
            raise ValueError('program_review_artifact_kind_mismatch')
        return json.loads(raw).get('contentItems', [])

    def _publish(self, entry, **changes):
        """Persist a coherent boundary snapshot and wake its main-actor caller."""
        from .program_review import trusted_summary
        with entry['condition']:
            job = entry['job']
            job.update(changes)
            job.update(calls=entry['calls'], model_wait_s=entry['model_wait_s'],
                       review_wait_s=entry['review_wait_s'],
                       action_summary=trusted_summary(self.store, entry['action_evidence']),
                       visual_feedback=deepcopy(entry.get('visual_feedback', [])),
                       final_observation_id=entry.get('final_observation_id'))
            job['action_summary']['review_rejections'] = sum(
                value == 'reject' for value in job.get('review_decisions', {}).values())
            if (job['action_summary'].get('status_counts') or {}).get('failed', 0) > 0:
                job['physical_action_failures_detected'] = True
            failures = entry.get('call_errors', [])
            job['call_summary'] = dict(source_kind='program_broker_tool_calls', calls=entry['calls'],
                failed_calls=len(failures), recent_errors=deepcopy(failures[-8:]))
            job['final_observation'] = (dict(observation_id=entry['final_observation_id'],
                source_kind='program_broker_tool_reply',
                delivery_semantics='Program access is not main-actor image delivery; the harness must forward this observation.')
                if entry.get('final_observation_id') else None)
            for name in ('stdout', 'stderr'):
                # Logs are host-owned, outside every model write root. Never
                # reopen a file in the program's mutable workspace as evidence.
                path = self.logs / (entry['job_id'] + '.' + name)
                try:
                    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                except FileNotFoundError:
                    data = b''
                else:
                    with os.fdopen(descriptor, 'rb') as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise ValueError('program_log_requires_regular_file')
                        data = stream.read()
                job[name+'_ref'] = self.store.put_artifact(data,
                    'program_'+name, 'text/plain')
            self.store.set_state('program', entry['job_id'], deepcopy(job))
            entry['condition'].notify_all()

    def _await_boundary(self, entry):
        with entry['condition']:
            while entry['job']['status'] not in TERMINAL | {'awaiting_review'}:
                entry['condition'].wait(.1)
        return self.status(entry['job_id'])

    @staticmethod
    def _continue_process(entry):
        process = entry.get('process')
        if process is not None and entry.get('process_paused'):
            try:
                os.killpg(process.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            entry['process_paused'] = False

    def _pause_process(self, entry):
        process = entry['process']
        os.killpg(process.pid, signal.SIGSTOP)
        entry['process_paused'] = True
        deadline = time.monotonic()+1
        try:
            while time.monotonic() < deadline:
                if entry['cancel'].is_set():
                    raise ValueError('program_channel_expired')
                stopped = os.waitid(os.P_PID, process.pid, os.WSTOPPED | os.WNOHANG)
                if stopped is not None and stopped.si_code == os.CLD_STOPPED:
                    return
                if process.poll() is not None:
                    raise ValueError('program_exited_before_review')
                time.sleep(.005)
            raise ValueError('program_review_pause_unconfirmed')
        except Exception:
            self._continue_process(entry)
            raise

    def _review_action(self, entry, arguments, correlation, identifier, record):
        from .program_review import reviewed_plan, review_execution_state
        from ...core.presentation.actor_presentation import ActorPresentation
        mode = arguments.get('mode', 'preview')
        preview_args = (dict(mode='inspect', plan_id=arguments.get('plan_id')) if mode == 'execute'
                        else {k: v for k, v in {**arguments, 'mode': 'preview'}.items() if k != 'auto_execute_if_clean'})
        preview_correlation = {**correlation, 'tool_call_id': identifier+'-preview'}
        self.store.event('program.call_started', {'tool': 'robot_action', 'arguments': preview_args},
            preview_correlation)
        preview = self._invoke('robot_action', preview_args, preview_correlation)
        plan = reviewed_plan(preview)
        review_id = 'review-' + digest([entry['job_id'], identifier, plan['plan_id']])[:24]
        gate = dict(review_id=review_id, plan_id=plan['plan_id'], operation_call_id=identifier,
                    preview=plan, preview_ref=self.store.put_artifact(
                        canonical(ActorPresentation(self.store).present_response(preview)).encode(),
                        'program_review_preview', 'application/json'),
                    kind='robot_action', decision='pending', semantics='Approve through program_run resume; do not execute the plan separately.')
        self._await_review(entry, gate, correlation)
        current = reviewed_plan(self._invoke('robot_action',
            dict(mode='inspect', plan_id=plan['plan_id']),
            {**correlation, 'tool_call_id': identifier+'-approved-inspect'}))
        if review_execution_state(current) != review_execution_state(plan):
            raise ValueError('program_action_plan_changed_since_review')
        # A second caller may advance even a zero-step checkpoint between inspect
        # and execute. The native cache checks cursor and revision atomically.
        execute_args = dict(mode='execute', plan_id=plan['plan_id'],
            expected_cursor=plan.get('executed_batch_native_steps', 0),
            expected_revision=plan.get('plan_revision', 0))
        if arguments.get('action_id'):
            execute_args['action_id'] = arguments['action_id']
        if entry['cancel'].is_set() or program_remaining(entry) <= 0:
            raise ValueError('program_channel_expired')
        record.update(action_dispatch_attempted=True, executed_arguments=execute_args)
        self.store.set_state('program_call',identifier,record)
        return self._invoke('robot_action', execute_args, correlation), execute_args

    def _review_wait_action(self, entry, arguments, correlation, identifier, record):
        frozen = deepcopy(arguments)
        preview = dict(kind='bounded_wait', status='review_required', request=frozen,
            observation_id=frozen.get('observation_id'),
            semantics='Hold measured arm positions and gripper goals while advancing the requested native steps; consumes native action budget.')
        response = dict(success=True, contentItems=[dict(type='inputText', text=canonical(preview))])
        review_id = 'review-' + digest([entry['job_id'], identifier, frozen])[:24]
        gate = dict(review_id=review_id, kind='robot_wait', operation_call_id=identifier,
            preview=preview, preview_ref=self.store.put_artifact(canonical(response).encode(),
                'program_review_preview','application/json'), decision='pending',
            semantics='Approve through program_run resume; the exact wait and original observation are retained.')
        self._await_review(entry, gate, correlation)
        if entry['cancel'].is_set() or program_remaining(entry) <= 0:
            raise ValueError('program_channel_expired')
        observed = self.services.observation()['observation_id']
        if observed is not None and observed != frozen.get('observation_id'):
            raise ValueError('stale_observation')
        record.update(action_dispatch_attempted=True, executed_arguments=frozen)
        self.store.set_state('program_call', identifier, record)
        return self._invoke('robot_wait', frozen, correlation), frozen

    def _await_review(self, entry, gate, correlation):
        with review_wait(entry):
            # Blocking the client RPC retains its Python stack; stopping its whole
            # process also prevents another Python thread from running during review.
            entry['process_ready'].wait()
            if entry['cancel'].is_set():
                raise ValueError('program_channel_expired')
            process = entry.get('process')
            if process is None or process.poll() is not None:
                raise ValueError('program_exited_before_review')
            with entry['condition']:
                if entry['cancel'].is_set():
                    raise ValueError('program_channel_expired')
                self._pause_process(entry)
                entry['review_event'].clear()
                entry['review'] = gate
                self._publish(entry, status='awaiting_review', pending_review=deepcopy(gate))
            self.store.event('program.review_requested', gate, correlation)
            entry['review_event'].wait()
            self._continue_process(entry)
            if entry['cancel'].is_set():
                raise ValueError('program_channel_expired')
            if gate['decision'] != 'approve':
                raise ValueError('program_action_review_rejected')

    def resume(self, identifier, review_id, decision, correlation):
        if correlation.get('program_id'):
            raise ValueError('program_tool_not_permitted')
        if decision not in {'approve', 'reject'}:
            raise ValueError('invalid_program_review_decision')
        saved = self.store.read_state('program', identifier)
        if not saved:
            raise ValueError('unknown_program_job')
        if saved.get('correlation', {}).get('thread_id') != correlation.get('thread_id'):
            raise ValueError('program_review_thread_mismatch')
        with self.lock:
            entry = self.entries.get(identifier)
        if entry is None:
            previous = saved.get('review_decisions', {}).get(review_id)
            if previous == decision:
                return self.status(identifier)
            raise ValueError('program_review_not_active')
        with entry['condition']:
            previous = entry['job'].get('review_decisions', {}).get(review_id)
            if previous:
                if previous != decision:
                    raise ValueError('program_review_decision_conflict')
                return self.status(identifier)
            gate = entry.get('review')
            if (entry['cancel'].is_set() or entry['job']['status'] != 'awaiting_review'
                    or not gate or gate['review_id'] != review_id):
                raise ValueError('program_review_not_active')
            gate['decision'] = decision
            entry['job'].setdefault('review_decisions', {})[review_id] = decision
            entry['job'].pop('pending_review', None)
            self._publish(entry, status='running')
            self.store.event('program.review_decided', dict(review_id=review_id, decision=decision,
                kind=gate['kind'], plan_id=gate.get('plan_id')), correlation)
            self._continue_process(entry)
            entry['review_event'].set()
        return self._await_boundary(entry)

    def broker(self, connection, entry):
        # Buffered reads avoid one syscall per byte for multi-megabyte geometry
        # artifacts; buffered writes plus flush deliver complete framed replies.
        try:
            with connection, connection.makefile("rwb") as pipe:
                while True:
                    line=pipe.readline(1024*1024+1)
                    if not line: return
                    try:
                        if len(line)>1024*1024: raise ValueError("program_request_too_large")
                        result={"ok":True,"result":self.dispatch(entry,json.loads(line))}
                    except Exception as error: result={"ok":False,"error":str(error)}
                    pipe.write((canonical(result)+"\n").encode())
                    pipe.flush()
        except OSError:
            # Process cancellation may reset reads or break the buffered close
            # flush as well as an explicit write. The supervisor owns its status.
            return

    def _validate_tool(self, name):
        from .program_review import LEGACY_ACTION_TOOLS
        if name in LEGACY_ACTION_TOOLS:
            raise ValueError('program_action_requires_robot_action')
        try:
            self.services.validate_tool(name)
        except ValueError:
            raise ValueError('program_tool_not_permitted') from None

    def _validate_arguments(self, name, arguments):
        self.services.validate_arguments(name, arguments)

    def _invoke(self, name, arguments, correlation):
        return self.services.invoke(name, arguments, correlation)

    def _tool_effect(self, name, arguments):
        return self.services.effect(name, arguments)

    def _requires_review(self, entry, tool, arguments):
        if entry.get('allow_direct_actions', False):
            return False
        if self._tool_effect(tool, arguments) != 'physical':
            return False
        if tool not in {'robot_wait', 'robot_action'}:
            raise ValueError('program_physical_tool_requires_direct_authorization')
        return True

    def dispatch(self, entry, value):
        from ...core.presentation.actor_presentation import ActorPresentation
        from ...core.tool_output import wire_response
        presentation = ActorPresentation(self.store)
        value = presentation.restore(value)
        if entry["cancel"].is_set() or program_remaining(entry) <= 0:
            raise ValueError("program_channel_expired")
        # One operation at a time also makes retry idempotency concrete.
        with entry['call_lock']:
            if entry["cancel"].is_set() or program_remaining(entry) <= 0:
                raise ValueError("program_channel_expired")
            if 'artifact_ref' in value:
                info, content = self.store.resolve_artifact(value['artifact_ref'],allowed_scopes=('actor','internal'))
                if len(content)>16*1024*1024: raise ValueError('program_artifact_too_large')
                profile = self.services.observation()['observation_profile']
                kind = info.get('kind', '')
                if kind.startswith('depth') and not allows_depth(profile):
                    raise ValueError('depth_artifact_not_permitted_by_observation_profile')
                # General scene/geometry artifacts may be actor-derived estimates.
                # Block known exact-calibration products, then inspect every JSON
                # body below; a broad kind prefix would reject legal RGB reasoning.
                if profile == LEGACY_ACTOR_PROFILE and kind in {
                        'calibration', 'camera_calibration', 'camera_pose_sample', 'robot_model',
                        'camera_poses', 'geometry_overlay', 'motion_preview_overlay'}:
                    raise ValueError('calibrated_artifact_not_permitted_by_observation_profile')
                mime = info.get('mime_type', '').split(';', 1)[0].strip().lower()
                if mime == 'application/json' or mime.endswith('+json'):
                    payload = json.loads(content)
                    if kind in {'unified_action_result', 'program_action_evidence'}:
                        from ...core.observation_policy import reject_recorded_action_privileged
                        records = payload if kind == 'program_action_evidence' and isinstance(payload, list) else [payload]
                        for record in records:
                            reject_recorded_action_privileged(record, self.store, profile)
                    else:
                        reject_privileged(payload, profile)
                    # Artifact kind is producer-owned; unknown JSON keeps its business fields.
                    projection = {'unified_action_result': 'action_result',
                        'program_action_evidence': 'program_action_evidence',
                        'manipulation_receipt': 'program_action_evidence',
                        'candidate_trajectory': 'owner_record'}.get(kind)
                    if projection:
                        from ...core.presentation.actor_output import project_metadata
                        content = canonical(project_metadata(payload, projection)).encode('utf-8')
                info, content = presentation.present_artifact(info, content)
                return {'artifact':info,'data_base64':base64.b64encode(content).decode()}
            operation = value.get('operation_id')
            if not isinstance(operation,str) or not operation or len(operation)>80: raise ValueError('invalid_program_operation_id')
            identifier = entry['job_id']+'-'+digest(operation)[:16]
            previous = self.store.read_state('program_call',identifier)
            if previous:
                if previous['input_digest']!=digest(value): raise ValueError('program_operation_content_conflict')
                if previous['status']!='completed': raise ValueError('program_operation_status_unknown_do_not_retry')
                # Older completed wire records may predate the current model
                # presentation policy. Project a copy without changing the
                # archived result or dispatching the operation again.
                return presentation.present(previous['result'])
            from .program_review import action_evidence, live_observation_id
            self._validate_tool(value.get('tool'))
            if not isinstance(value.get('arguments'),dict): raise ValueError('program_arguments_require_object')
            entry['calls'] += 1
            record = {'program_id':entry['job_id'],'operation_id':operation,'input_digest':digest(value),'status':'dispatching'}
            self.store.set_state('program_call',identifier,record)
            correlation = {**entry['correlation'],'program_id':entry['job_id'],'tool_call_id':identifier}
            self.store.event('program.call_started',{'tool':value['tool'],'arguments':value['arguments']},correlation)
            try:
                arguments = value['arguments']
                self._validate_arguments(value['tool'], arguments)
                arguments = self._action_timeout(entry, value['tool'], arguments)
                if self._requires_review(entry, value['tool'], arguments):
                    review = self._review_wait_action if value['tool'] == 'robot_wait' else self._review_action
                    raw, executed_arguments = review(entry, arguments, correlation, identifier, record)
                else:
                    if self._tool_effect(value['tool'], arguments) == 'physical':
                        record.update(action_dispatch_attempted=True, executed_arguments=arguments)
                        self.store.set_state('program_call',identifier,record)
                    raw = self._invoke(value['tool'],arguments,correlation)
                    executed_arguments = arguments
                contract_error = getattr(raw, 'contract_error', None)
                if contract_error is not None:
                    record['output_contract_error'] = deepcopy(contract_error)
                    entry.setdefault('call_errors', []).append(dict(call_id=identifier,
                        tool=value['tool'], reason=contract_error['code'], phase='after_handler'))
                evidence = action_evidence(self.store, value['tool'], executed_arguments, raw, identifier,
                    physical=bool(record.get('action_dispatch_attempted')))
                if evidence is not None:
                    entry.setdefault('action_evidence', []).append(evidence)
                    if evidence.get('physics_advanced') is not False:
                        entry['final_observation_id'] = evidence.get('result_observation_id')
                        if entry.get('visual_feedback'):
                            # 物理动作执行后已有最新末帧观测，清除此前未执行的候选 preview 叠加图
                            entry['visual_feedback'][:] = [
                                item for item in entry['visual_feedback']
                                if not (item.get('tool') == 'robot_action'
                                        and item.get('arguments', {}).get('mode', 'preview') == 'preview')
                            ]
                observed = live_observation_id(value['tool'], raw) if not arguments.get('camera_id') else None
                if observed:
                    entry['final_observation_id'] = observed
                result = presentation.present_program_response(raw)
                # Action/current-observation replies already have a final-frame path.
                # Keep other visual replies (including previews and historical crops).
                if evidence is None and not observed and any(
                        item.get('type') == 'inputImage' for item in result.get('contentItems', [])):
                    visuals = entry.setdefault('visual_feedback', [])
                    if value['tool'] == 'robot_action' and arguments.get('mode', 'preview') == 'preview':
                        # 候选搜索循环中仅保留最新一次 robot_action preview 的叠加图，避免向主模型重复推送已淘汰候选图
                        visuals[:] = [item for item in visuals
                                      if not (item.get('tool') == 'robot_action'
                                              and item.get('arguments', {}).get('mode', 'preview') == 'preview')]
                    visuals.append(dict(call_id=identifier,
                        tool=value['tool'], arguments=deepcopy(arguments),
                        reply_ref=self.store.put_artifact(canonical(result).encode(),
                            'program_visual_reply', 'application/json')))
                record.update(status='completed',result=result)
                self.store.set_state('program_call',identifier,record)
                if entry.get('visual_feedback') and entry['visual_feedback'][-1]['call_id'] == identifier:
                    self._publish(entry)
                self.store.event('program.call_completed',{'tool':value['tool'],'operation_id':operation},correlation)
                return result
            except Exception as error:
                from ...core.tool_errors import error_payload
                from ...core.tool_output import ToolResponse, StructuredText
                entry.setdefault('call_errors', []).append(dict(call_id=identifier,
                    tool=value.get('tool'), reason=str(error)[:512]))
                failure = error_payload(error, tool=value.get('tool'), arguments=value.get('arguments'))
                record.update(status='completed', result=presentation.present_program_response(
                    ToolResponse([StructuredText(failure)], data=failure, kind='error', success=False)))
                if record.get('action_dispatch_attempted'):
                    evidence = action_evidence(self.store, value['tool'], record['executed_arguments'],
                        record['result'], identifier, physical=True)
                    if evidence is not None and (not entry.get('action_evidence')
                            or entry['action_evidence'][-1]['call_id'] != identifier):
                        entry.setdefault('action_evidence', []).append(evidence)
                        entry['final_observation_id'] = evidence.get('result_observation_id')
                self.store.set_state('program_call',identifier,record)
                self.store.event('program.call_failed',{'tool':value['tool'],'operation_id':operation,
                    'reason':str(error)},correlation)
                return record['result']

    def cancel(self, identifier):
        with self.lock:
            entry = self.entries.get(identifier)
        if entry:
            entry['cancel'].set()
            if entry.get('review_event'):
                with entry['condition']:
                    self._continue_process(entry)
                    entry['review_event'].set()
            self.services.cancel_action(identifier)
        return self.status(identifier)

    def run(self, arguments, correlation):
        if correlation.get('program_id'):
            raise ValueError('program_tool_not_permitted')
        operation = arguments.get('operation', 'start')
        if operation == 'resume':
            if set(arguments) - {'operation', 'job_id', 'review_id', 'decision'}:
                raise ValueError('program_resume_cannot_change_invocation')
            return self.resume(arguments.get('job_id'), arguments.get('review_id'),
                               arguments.get('decision'), correlation)
        if operation != 'start':
            raise ValueError('invalid_program_operation')
        source = arguments.get('source')
        if source is None and arguments.get('source_ref'):
            info, content = self.store.resolve_artifact(arguments['source_ref'],allowed_scopes=('actor',))
            if info['kind']!='program_source': raise ValueError('program_source_ref_kind_mismatch')
            source=content.decode()
        timeout = arguments.get('timeout_s',180)
        allow_direct = arguments.get('allow_direct_actions', False)
        if not isinstance(source,str) or not source.strip() or len(source.encode())>128*1024: raise ValueError('invalid_program_source')
        if type(timeout) not in (int,float) or not 1<=timeout<=300: raise ValueError('invalid_program_budget')
        if type(allow_direct) is not bool: raise ValueError('invalid_program_action_authorization')
        # max_calls is accepted for old clients, but is never an execution limit.
        if 'max_calls' in arguments and (type(arguments['max_calls']) is not int or arguments['max_calls']<1):
            raise ValueError('invalid_program_budget')
        identifier = 'program-'+digest([correlation['thread_id'],correlation['turn_id'],correlation['tool_call_id']])[:24]
        existing = self.store.read_state('program',identifier)
        if existing:
            if existing['input_digest']!=digest(arguments): raise ValueError('program_id_content_conflict')
            return self.status(identifier)
        from ...core.presentation.actor_presentation import ActorPresentation
        source = ActorPresentation(self.store).present_identifiers(source)
        workspace = self.root / identifier
        workspace.mkdir()
        (workspace/'main.py').write_text(source)
        shutil.copyfile(Path(__file__).with_name('auriga_client.py'),workspace/'auriga_client.py')
        for module in self.services.workspace_modules:
            shutil.copyfile(Path(__file__).parents[2] / 'perception' / 'external' / module, workspace/module)
            if module == 'scene3d_helpers.py':
                vf_path = Path(__file__).parents[2] / 'perception' / 'external' / 'virtual_field.py'
                if vf_path.is_file():
                    shutil.copyfile(vf_path, workspace / 'virtual_field.py')
        source_ref = self.store.put_artifact(source.encode(),'program_source','text/x-python')
        job = {'job_id':identifier,'identity':deepcopy(self.store.identity),'status':'running','source_ref':source_ref,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'input_digest':digest(arguments),'correlation':correlation,'created_at':utc_now(),'timeout_s':timeout,'timeout_semantics':'work_wall_time_excluding_own_model_and_main_review_waits','call_limit':None,'allow_direct_actions':allow_direct,
               'action_authorization_scope':'this_program_invocation_only','review_decisions':{}}
        self.store.set_state('program',identifier,job)
        self.store.event('program.started',job,correlation)
        entry = {'job_id':identifier,'correlation':correlation,'cancel':threading.Event(),'call_lock':threading.Lock(),'calls':0,
                 'deadline':time.monotonic()+timeout,'budget_lock':threading.RLock(),'model_wait_started_at':None,'model_wait_s':0.0,
                 'review_wait_started_at':None,'review_wait_s':0.0,'allow_direct_actions':allow_direct,
                 'condition':threading.Condition(threading.RLock()),'review_event':threading.Event(),
                 'process_ready':threading.Event(),'process_paused':False,'workspace':workspace,'job':job,'action_evidence':[]}
        with self.lock: self.entries[identifier]=entry
        worker = threading.Thread(target=self._execute_guarded, args=(entry,), daemon=True)
        entry['worker'] = worker
        worker.start()
        return self._await_boundary(entry)

    def _execute_guarded(self, entry):
        try:
            self._execute(entry)
        except Exception as error:
            # Includes failures before Popen (socket/spec/log setup). Never leave
            # the main actor waiting forever on a supervisor which has exited.
            entry['cancel'].set()
            entry['process_ready'].set()
            entry['review_event'].set()
            self._continue_process(entry)
            for channel in entry.get('channel_sockets', ()):
                try: channel.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                channel.close()
            broker = entry.get('broker')
            if broker:
                broker.join(1)
            process = entry.get('process')
            stopped = process is None or process.poll() is not None
            if not stopped:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(2)
                    stopped = True
                except (OSError, subprocess.TimeoutExpired):
                    pass
            active = self.services.cancel_action(entry['job_id'])
            stopped = stopped and not active and not (broker and broker.is_alive())
            self._publish(entry, status='failed' if stopped else 'unknown', reason=str(error),
                          stop_confirmed=bool(stopped), finished_at=utc_now())
            with self.lock:
                self.entries.pop(entry['job_id'], None)

    def _execute(self, entry):
        """Own the subprocess until terminal acknowledgement, across review yields."""
        identifier, workspace, job = entry['job_id'], entry['workspace'], entry['job']
        correlation = entry['correlation']
        parent,child=socket.socketpair()
        entry['channel_sockets'] = (parent, child)
        broker=threading.Thread(target=self.broker,args=(parent,entry),daemon=True)
        entry['broker'] = broker
        broker.start()
        if self.services.sandbox_enabled:
            with self.runtime_lock:
                if self.runtime is None:
                    self.runtime = program_runtime(self.store.root / 'private' / 'program-runtime',
                        sandbox_enabled=self.services.sandbox_enabled)
                runtime, packages, read_paths = self.runtime
            env = {'PATH':os.defpath,'HOME':str(workspace),'TMPDIR':str(workspace),
                   'PYTHONHOME':str(runtime),'PYTHONPATH':str(workspace)+os.pathsep+str(packages),
                   'PYTHONDONTWRITEBYTECODE':'1','AURIGA_PROGRAM_FD':str(child.fileno())}
        else:
            read_paths = []
            py_path = str(workspace)
            if os.environ.get('PYTHONPATH'):
                py_path = py_path + os.pathsep + os.environ['PYTHONPATH']
            env = {'PATH':os.defpath,'HOME':str(workspace),'TMPDIR':str(workspace),
                   'PYTHONPATH':py_path,
                   'PYTHONDONTWRITEBYTECODE':'1','AURIGA_PROGRAM_FD':str(child.fileno())}
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', OPENCV_FOR_THREADS_NUM='1')
        limits = {'RLIMIT_FSIZE':8*1024*1024,'RLIMIT_AS':1024*1024*1024}
        spec = self.store.root/'private'/('isolation-'+identifier+'.json')
        spec.parent.mkdir(exist_ok=True)
        spec.write_text(canonical({'sandbox_enabled':self.services.sandbox_enabled,'read_paths':read_paths,'write_paths':[str(workspace),'/dev/null','/dev/urandom'],'tcp_ports':[],'deny_socket_creation':True,'bounded_program':True,'limits':limits,'env':env}))
        spec.chmod(0o600)
        output, errors = self.logs/(identifier+'.stdout'), self.logs/(identifier+'.stderr')
        process = None
        def _apply_limits():
            for name, value in limits.items():
                limit = getattr(resource, name, None)
                if limit is not None:
                    _, hard = resource.getrlimit(limit)
                    cap = int(value) if hard == resource.RLIM_INFINITY else min(int(value), hard)
                    resource.setrlimit(limit, (cap, cap))
        try:
            with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stdout, \
                    os.fdopen(os.open(errors, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as stderr:
                if self.services.sandbox_enabled:
                    process = subprocess.Popen([sys.executable,str(Path(__file__).with_name('isolation.py')),str(spec),sys.executable,'-S',str(workspace/'main.py')],cwd=Path(__file__).resolve().parents[3],stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,pass_fds=(child.fileno(),))
                else:
                    process = subprocess.Popen([sys.executable,str(workspace/'main.py')],cwd=str(workspace),env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,pass_fds=(child.fileno(),),preexec_fn=_apply_limits)
                self.last_activity_at = time.monotonic()
                output_sizes = (0, 0)
                child.close()
                entry['process']=process
                entry['process_ready'].set()
                while process.poll() is None:
                    sizes = (os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                    if sizes != output_sizes:
                        self.last_activity_at = time.monotonic()
                        output_sizes = sizes
                    if entry['cancel'].is_set() or program_remaining(entry) <= 0:
                        entry['cancel'].set()
                        self.cancel(identifier)
                        self._continue_process(entry)
                        try: os.killpg(process.pid,signal.SIGTERM)
                        except ProcessLookupError: pass
                        try: process.wait(2)
                        except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                        break
                    time.sleep(.02)
                self.last_activity_at = time.monotonic()
                # Kill descendants even after a parent exits: no detached actor jobs.
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                # A cancelled action owns a separate backend operation; wait for its acknowledgement.
                deadline=time.monotonic()+45
                while self.services.action_active(identifier) and time.monotonic()<deadline: time.sleep(.05)
                unknown=self.services.action_active(identifier)
                final_status='unknown' if unknown else ('cancelled' if entry['cancel'].is_set() else 'succeeded' if process.returncode==0 else 'failed')
                job.update(returncode=process.returncode,stop_confirmed=not unknown,finished_at=utc_now())
        except Exception as error:
            final_status='unknown' if process and process.poll() is None else 'failed'
            job.update(reason=str(error),stop_confirmed=not bool(process and process.poll() is None),finished_at=utc_now())
        finally:
            child.close()
            entry['cancel'].set()
            entry['process_ready'].set()
            self._continue_process(entry)
            entry['review_event'].set()
            try: parent.shutdown(socket.SHUT_RD)
            except OSError: pass
            broker.join(45)
            if broker.is_alive():
                final_status='unknown'
                job.update(stop_confirmed=False,reason='program_broker_operation_stop_unconfirmed')
            job.pop('pending_review', None)
            self._publish(entry, status=final_status)
            self.store.event('program.'+final_status,deepcopy(job),correlation)
            with self.lock: self.entries.pop(identifier,None)

    def close(self):
        with self.lock:
            entries = list(self.entries.values())
        self.cancel_all()
        deadline = time.monotonic()+50
        for entry in entries:
            worker = entry.get('worker')
            if worker and worker is not threading.current_thread():
                worker.join(max(0.0, deadline-time.monotonic()))
        return self.snapshot()
