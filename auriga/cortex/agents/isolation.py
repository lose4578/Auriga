"""Shared isolation and pinning helpers for the Claude Code and OpenCode agent backends (design rev 3, section 8).

Nothing here talks to a model or imports an agent SDK.  The helpers are deliberately small and explicit:

* ``clean_environment``      build the agent environment from explicit inputs, never from ``os.environ``;
* ``private_dirs``           create 0700 per-run directories and refuse symlinks;
* ``write_env_launcher``     write an ``env -i`` launcher script (the Claude Code SDK merges ``options.env`` over the
                             inherited environment, so a launcher passed as ``cli_path`` is the only way to start
                             from an empty environment; same technique as ``codex_evidence_agent``);
* ``file_sha256_cached``     sha256 of an executable, cached by (real path, size, mtime);
* ``require_exact_version``  hard error ``<name>_version_mismatch`` on any difference from the pin;
* ``process_tree_snapshot`` / ``terminate_tree``  teardown that also finds descendants which left the process group
  (OpenCode bash-tool children are detached with ``setsid`` and survive a plain ``killpg``, spike S8);
* ``AGENT_MARKER_ENV`` / ``processes_with_marker`` / ``terminate_agent_processes``  a per-session process marker in
  the agent environment (inherited by every native tool child) and the teardown that also finds and verifies the
  children that left the agent's process tree entirely (re-parented to pid 1 after their shell exited).

Teardown order for callers (design 3.7): abort the agent's running tools -> refresh the known tree while the root
is still alive -> ``stop_agent_tree(root_pid, alive, known, instance_id, cwd_roots=[actor dir])`` (re-validated tree
plus marked processes, killed, then re-scanned) -> record ``report.confirmed`` (else
``agent_process_exit_unconfirmed``). Both MCP backends use exactly this path.

What the marker can see (verified on macOS 26.6 with SIP, uid 501, and by reading Linux procfs rules):

* Linux: ``/proc/<pid>/environ`` of every same-uid, dumpable process (not setuid/setgid programs).
* macOS: ``sysctl(KERN_PROCARGS2)`` returns the environment of same-uid processes EXCEPT Apple platform binaries
  (``/bin/sh``, ``/bin/zsh``, ``/usr/bin/perl``, ``/bin/sleep`` ...), whose environment is withheld (0 entries; ``ps
  -E`` shows nothing either). For a process whose environment cannot be read, the scan falls back to its working
  directory (``proc_pidinfo(PROC_PIDVNODEPATHINFO)``, readable for every same-uid process): it is marked when its
  cwd lies inside one of the session's agent directories (``cwd_roots``, the actor dir) AND it is orphaned (parent
  pid 1) or a child of a marked process, so an operator's shell that merely ``cd``-ed into the run is never selected.

Residuals (documented, not mitigated, design rev 3.2): the agent's own code can drop the marker (``env -u``,
``env -i``) and, on macOS, a platform binary started that way and running outside the agent directories is not
found; the marker value is visible to the agent's shell (it is an opaque session id, no credential).
"""
from __future__ import annotations

import hashlib
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "VersionMismatch", "ProcessInfo", "TreeSnapshot", "TerminationReport", "clean_environment", "private_dirs",
    "write_env_launcher", "file_sha256_cached", "require_exact_version", "process_tree_snapshot", "terminate_tree",
    "parse_ps_table", "parse_proc_stat", "build_snapshot", "process_table", "teardown_snapshot",
    "AGENT_MARKER_ENV", "MarkedProcess", "MarkerScan", "AgentTeardownReport", "agent_marker",
    "processes_with_marker", "select_marked", "terminate_agent_processes", "stop_agent_tree",
]

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# Variables copied from ``base`` are dropped when their name looks like a host credential or a host-service endpoint.
_CREDENTIAL_NAME = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION|CERT|BEARER)"
    r"|^(ANTHROPIC|OPENAI|CODEX|CLAUDE|OPENCODE|AWS|AZURE|GOOGLE|GCP|GCLOUD|GITHUB|GITLAB|HF|HUGGING|"
    r"SSH|GPG|NPM|DOCKER|KUBE|VERTEX|BEDROCK|OPENROUTER|MISTRAL|COHERE|GROQ|XAI)_"
    r"|(^|_)(PROXY)$|^SSH_AUTH_SOCK$", re.IGNORECASE)
_ALWAYS_EXPLICIT = frozenset({"PATH", "HOME", "TMPDIR"})
_ENV_EXECUTABLE = "/usr/bin/env"
_PS_EXECUTABLE = "/bin/ps"


class VersionMismatch(RuntimeError):
    """Pinned-version check failed.  ``str(error)`` is exactly ``<name>_version_mismatch``."""

    def __init__(self, name: str, actual: object, pinned: str):
        super().__init__(f"{name}_version_mismatch")
        self.name, self.actual, self.pinned = name, actual, pinned


# --------------------------------------------------------------------------------------------- environment

def _check_env_item(key: object, value: object) -> None:
    if not isinstance(key, str) or not _ENV_NAME.match(key):
        raise ValueError(f"invalid_environment_name:{key!r}")
    if not isinstance(value, str) or "\0" in value:
        raise ValueError(f"invalid_environment_value:{key}")


def clean_environment(base: Mapping[str, str], *, path_dirs: Sequence[str | os.PathLike] = (),
                      home: str | os.PathLike, tmpdir: str | os.PathLike,
                      extra: Mapping[str, str] | None = None, instance_id: str | None = None) -> dict[str, str]:
    """Return the complete environment for an agent process.  Reads no global state (never ``os.environ``).

    Parameters
    ----------
    base:
        Caller-chosen seed variables (e.g. ``{"LANG": "C.UTF-8", "TZ": "UTC"}``).  Entries whose name looks like a host
        credential or host endpoint (``*KEY*``, ``*TOKEN*``, ``*SECRET*``, ``*PASSWORD*``, ``*AUTH*``, ``*PROXY``,
        ``ANTHROPIC_*``, ``OPENAI_*``, ``AWS_*``, ``SSH_AUTH_SOCK`` ...) are dropped, and ``PATH``/``HOME``/``TMPDIR``
        are ignored because they are set from the explicit arguments.  Pass ``{}`` for none.
    path_dirs:
        Directories placed in front of ``os.defpath`` in ``PATH`` (e.g. a private bin dir holding ``rg``).
        Absolute paths only; default: none, so ``PATH`` is ``os.defpath``.
    home:
        ``HOME`` of the agent (design: ``<run>/actor``).  Absolute path.
    tmpdir:
        ``TMPDIR`` of the agent (design: ``<run>/private/tmp``).  Absolute path.
    extra:
        Backend variables applied last and unfiltered (relay URL, placeholder credentials, feature flags).  Default none.
    instance_id:
        The agent session's instance id (``runtime/agent.instance_id``).  When given, ``AGENT_MARKER_ENV`` is set to
        it (after ``extra``), so every process the agent starts carries the marker that ``processes_with_marker``
        finds at teardown.  Default None: no marker.

    ``LANG`` defaults to ``C.UTF-8`` unless ``base`` supplies it.  Keys must be valid shell names and values must not
    contain NUL; otherwise ``ValueError``.
    """
    dirs = [os.fspath(item) for item in path_dirs]
    for item in (*dirs, os.fspath(home), os.fspath(tmpdir)):
        if not os.path.isabs(item) or "\0" in item:
            raise ValueError(f"environment_path_must_be_absolute:{item}")
    if any(":" in item for item in dirs):
        raise ValueError("environment_path_dir_contains_colon")
    environment: dict[str, str] = {}
    for key, value in base.items():
        _check_env_item(key, value)
        if key in _ALWAYS_EXPLICIT or _CREDENTIAL_NAME.search(key):
            continue
        environment[key] = value
    environment.setdefault("LANG", "C.UTF-8")
    environment["PATH"] = ":".join([*dirs, os.defpath])
    environment["HOME"] = os.fspath(home)
    environment["TMPDIR"] = os.fspath(tmpdir)
    for key, value in (extra or {}).items():
        _check_env_item(key, value)
        environment[key] = value
    if instance_id is not None:
        environment[AGENT_MARKER_ENV] = agent_marker(instance_id)
    return environment


# ------------------------------------------------------------------------------------------ private dirs

def private_dirs(run_dir: str | os.PathLike, names: Iterable[str]) -> dict[str, Path]:
    """Create per-run private directories (mode 0700) below ``run_dir`` and return ``{name: path}``.

    ``run_dir`` must already exist and be a directory (it may itself be reached through a symlink; it is resolved).
    Each ``name`` is a relative path such as ``"tmp"`` or ``"opencode/xdg"``; ``..``, absolute names and empty
    components are rejected.  Every component below ``run_dir`` is created 0700 (the umask is overridden by an
    explicit ``chmod``), pre-existing components are re-chmodded to 0700, and a symlink or non-directory at any
    component raises ``ValueError("private_dir_symlink:...")`` / ``("private_dir_not_directory:...")``.
    Components must be owned by the current user.
    """
    root = Path(run_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"private_dir_not_directory:{root}")
    result: dict[str, Path] = {}
    for name in names:
        parts = Path(name).parts
        if not parts or Path(name).is_absolute() or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"private_dir_invalid_name:{name!r}")
        current = root
        for part in parts:
            current = current / part
            try:
                info = os.lstat(current)
            except FileNotFoundError:
                os.mkdir(current, 0o700)
                info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"private_dir_symlink:{current}")
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"private_dir_not_directory:{current}")
            if info.st_uid != os.getuid():
                raise ValueError(f"private_dir_foreign_owner:{current}")
            if stat.S_IMODE(info.st_mode) != 0o700:
                os.chmod(current, 0o700)
        result[name] = current
    return result


# ------------------------------------------------------------------------------------------ env launcher

def write_env_launcher(path: str | os.PathLike, real_executable: str | os.PathLike, env: Mapping[str, str], *,
                       reexport: Iterable[str] = ()) -> Path:
    """Write a POSIX ``sh`` launcher ``exec /usr/bin/env -i KEY=VALUE... <real_executable> "$@"`` and return its path.

    Parameters
    ----------
    path:
        Launcher file to create (mode 0700, replaced if it exists; a symlink at ``path`` is refused).  Pass it as
        the SDK ``cli_path``; record the *real* executable and its sha256 as provenance, not the launcher.
    real_executable:
        Absolute path of the real CLI.  Not resolved or checked here.
    env:
        The complete child environment (from ``clean_environment``).  Values are single-quote escaped, so spaces,
        quotes, ``$``, backticks and newlines are preserved literally.
    reexport:
        Names of variables the caller's SDK sets *at spawn time* that ``env -i`` would drop (Claude Code:
        ``CLAUDE_CODE_ENTRYPOINT``, ``CLAUDE_AGENT_SDK_VERSION``, ``CLAUDE_CODE_SDK_READS_SESSION_STATE``).  At run
        time each one is copied from the launcher's own environment, only if set there.  A name also present in
        ``env`` is an error (ambiguous precedence).  Default none.
    """
    executable = os.fspath(real_executable)
    if not os.path.isabs(executable) or "\0" in executable:
        raise ValueError("launcher_executable_must_be_absolute")
    names = list(reexport)
    for key, value in env.items():
        _check_env_item(key, value)
    for name in names:
        if not isinstance(name, str) or not _ENV_NAME.match(name):
            raise ValueError(f"invalid_reexport_name:{name!r}")
        if name in env:
            raise ValueError(f"reexport_name_in_env:{name}")
    quote = _sh_quote
    lines = ["#!/bin/sh", "# generated by auriga.cortex.agents.isolation; do not edit",
             "set -- " + quote(executable) + ' "$@"']
    for name in names:
        lines.append(f'if [ "${{{name}+set}}" = set ]; then set -- "{name}=${name}" "$@"; fi')
    if env:
        lines.append("set -- " + " ".join(quote(f"{key}={value}") for key, value in env.items()) + ' "$@"')
    lines.append(f'exec {_ENV_EXECUTABLE} -i "$@"')
    target = Path(path)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o700)
    except OSError as error:
        raise ValueError(f"launcher_path_unwritable:{target}:{error.strerror}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o700)
        handle.write("\n".join(lines) + "\n")
    return target


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------------------- pinning helpers

_SHA_CACHE: dict[tuple[str, int, int], str] = {}
_SHA_LOCK = threading.Lock()


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_cached(path: str | os.PathLike) -> str:
    """Lowercase hex sha256 of the file at ``path`` (symlinks resolved), cached by (real path, size, mtime_ns).

    A changed size or modification time re-hashes; unchanged files (large CLI binaries) are hashed once per
    process.  Raises ``FileNotFoundError`` for a missing file.  Thread-safe.
    """
    real = os.path.realpath(path)
    info = os.stat(real)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not_a_regular_file:{real}")
    key = (real, info.st_size, info.st_mtime_ns)
    with _SHA_LOCK:
        cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    digest = _hash_file(real)
    with _SHA_LOCK:
        for stale in [item for item in _SHA_CACHE if item[0] == real]:
            del _SHA_CACHE[stale]
        _SHA_CACHE[key] = digest
    return digest


def require_exact_version(name: str, actual: object, pinned: str) -> str:
    """Return ``actual`` (stripped) if it equals ``pinned`` exactly, else raise ``VersionMismatch``.

    ``name`` is the backend label (``"claude_code"``, ``"opencode"``); the error text is ``<name>_version_mismatch``
    with ``.actual`` / ``.pinned`` attributes for the cleanup/provenance record.  No prefix, range or ``v``-stripping:
    the caller passes the bare version token (e.g. ``"2.1.321"``).  Never falls back to a different binary.
    """
    if not isinstance(name, str) or not _ENV_NAME.match(name):
        raise ValueError(f"invalid_version_check_name:{name!r}")
    text = actual.strip() if isinstance(actual, str) else None
    if text is None or text != pinned:
        raise VersionMismatch(name, actual, pinned)
    return text


# ------------------------------------------------------------------------------------ process tree teardown

@dataclass(frozen=True)
class ProcessInfo:
    """One process row: ``pid``, parent ``ppid`` and process-group id ``pgid`` (all ints)."""
    pid: int
    ppid: int
    pgid: int


@dataclass(frozen=True)
class TreeSnapshot:
    """Root pid plus every descendant found at snapshot time (root first, parents before children).

    ``members`` is empty when the root was already gone; descendants of an already-dead root are re-parented to
    init and can no longer be found, so snapshot *before* stopping the root.
    """
    root_pid: int
    members: tuple[ProcessInfo, ...]

    @property
    def pgids(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(item.pgid for item in self.members))


@dataclass
class TerminationReport:
    """Typed result of ``terminate_tree``.

    ``confirmed`` is True only if every snapshot member was verified gone.  ``survivors`` lists pids still alive
    after SIGKILL and the verify window (caller records ``agent_process_exit_unconfirmed``).  ``errors`` holds
    ``"<operation>:<pid-or-pgid>:<errno name>"`` strings for unexpected signal failures (ESRCH is not an error).
    """
    root_pid: int
    members: tuple[int, ...]
    signalled_groups: tuple[int, ...] = ()
    signalled_pids: tuple[int, ...] = ()
    exited_after_term: tuple[int, ...] = ()
    killed: tuple[int, ...] = ()
    survivors: tuple[int, ...] = ()
    skipped_self: tuple[int, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)
    elapsed_s: float = 0.0

    @property
    def confirmed(self) -> bool:
        return not self.survivors

    def to_dict(self) -> dict:
        return {"root_pid": self.root_pid, "members": list(self.members), "confirmed": self.confirmed,
                "signalled_groups": list(self.signalled_groups), "signalled_pids": list(self.signalled_pids),
                "exited_after_term": list(self.exited_after_term), "killed": list(self.killed),
                "survivors": list(self.survivors), "skipped_self": list(self.skipped_self),
                "errors": list(self.errors), "elapsed_s": round(self.elapsed_s, 3)}


def parse_ps_table(output: str) -> list[ProcessInfo]:
    """Parse ``ps -axo pid=,ppid=,pgid=`` output (whitespace separated ints; malformed lines are skipped)."""
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and all(item.lstrip("-").isdigit() for item in fields[:3]):
            rows.append(ProcessInfo(int(fields[0]), int(fields[1]), int(fields[2])))
    return rows


def parse_proc_stat(pid: int, text: str) -> ProcessInfo | None:
    """Parse one Linux ``/proc/<pid>/stat`` line.  The command field is parenthesised and may contain spaces and
    ``)``, so fields are split after the *last* ``)``: ``state ppid pgrp ...``.  Returns None if malformed."""
    end = text.rfind(")")
    if end < 0:
        return None
    rest = text[end + 1:].split()
    if len(rest) < 3 or not (rest[1].isdigit() and rest[2].isdigit()):
        return None
    return ProcessInfo(pid, int(rest[1]), int(rest[2]))


def _read_proc_table(proc_root: str | os.PathLike = "/proc") -> list[ProcessInfo]:
    rows = []
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        try:
            text = Path(proc_root, entry, "stat").read_text(errors="replace")
        except OSError:
            continue  # exited while scanning
        info = parse_proc_stat(int(entry), text)
        if info is not None:
            rows.append(info)
    return rows


def _read_ps_table() -> list[ProcessInfo]:
    completed = subprocess.run([_PS_EXECUTABLE, "-axo", "pid=,ppid=,pgid="], capture_output=True, text=True,
                               timeout=10, check=True, env={"PATH": os.defpath, "LC_ALL": "C"})
    return parse_ps_table(completed.stdout)


def build_snapshot(root_pid: int, table: Iterable[ProcessInfo]) -> TreeSnapshot:
    """Select ``root_pid`` and all its transitive children (by ppid) from a process table, regardless of pgid."""
    children: dict[int, list[ProcessInfo]] = {}
    by_pid: dict[int, ProcessInfo] = {}
    for row in table:
        by_pid[row.pid] = row
        children.setdefault(row.ppid, []).append(row)
    if root_pid not in by_pid:
        return TreeSnapshot(root_pid, ())
    ordered, seen, queue = [], {root_pid}, [by_pid[root_pid]]
    while queue:
        row = queue.pop(0)
        ordered.append(row)
        for child in sorted(children.get(row.pid, ()), key=lambda item: item.pid):
            if child.pid not in seen:
                seen.add(child.pid)
                queue.append(child)
    return TreeSnapshot(root_pid, tuple(ordered))


def process_table() -> list[ProcessInfo]:
    """Every process of the host as ``ProcessInfo`` rows (``/proc`` on Linux, ``/bin/ps`` elsewhere)."""
    return _read_proc_table() if sys.platform.startswith("linux") else _read_ps_table()


def teardown_snapshot(root_pid: int, root_alive: bool, known: Iterable[ProcessInfo],
                      table: Iterable[ProcessInfo]) -> TreeSnapshot:
    """The processes to stop at close, as a ``TreeSnapshot`` re-validated against the current ``table``.

    root_pid: the agent process pid; included only while ``root_alive`` (a dead root's pid may be reused).
    known: every ``ProcessInfo`` seen in earlier snapshots of the agent's tree (descendants that outlived their parent
    are re-parented and no longer reachable from the root). A known member is kept only while it is still the same
    process: same pid and same process group, and, inside the caller's own group, not a direct child of the caller (a
    recycled pid there would be the caller's own new child). Every member of a known process group other than the
    caller's is added (a detached tool shell survives in its own group), then all descendants of the selection. The
    caller's own process is never selected. Same rule as the Claude Code backend's teardown (R4-3).
    """
    table = list(table)
    me, own_group = os.getpid(), os.getpgrp()
    by_pid = {row.pid: row for row in table}
    selected: dict[int, ProcessInfo] = {}
    if root_alive and root_pid in by_pid:
        selected[root_pid] = by_pid[root_pid]
    groups = set()
    for member in known:
        if member.pid in (root_pid, me):
            continue
        row = by_pid.get(member.pid)
        if row is not None and row.pgid == member.pgid and (row.pgid != own_group or row.ppid != me):
            selected[member.pid] = row
        if member.pgid > 1 and member.pgid != own_group:
            groups.add(member.pgid)
    for row in table:
        if row.pgid in groups and row.pid != me:
            selected.setdefault(row.pid, row)
    children: dict[int, list[ProcessInfo]] = {}
    for row in table:
        children.setdefault(row.ppid, []).append(row)
    queue = list(selected)
    while queue:
        for child in children.get(queue.pop(), ()):
            if child.pid not in selected and child.pid != me:
                selected[child.pid] = child
                queue.append(child.pid)
    ordered = ([selected.pop(root_pid)] if root_pid in selected else []) + sorted(selected.values(),
                                                                                  key=lambda row: row.pid)
    return TreeSnapshot(root_pid, tuple(ordered))


def process_tree_snapshot(root_pid: int) -> TreeSnapshot:
    """Snapshot ``root_pid`` and every descendant, including ones that left the process group via ``setsid``.

    Linux reads ``/proc/<pid>/stat``; macOS (and other POSIX) runs ``/bin/ps -axo pid=,ppid=,pgid=``.  A process
    that forks and exits between snapshot and kill is not seen; call this as late as possible before
    ``terminate_tree`` and after aborting the agent's running tools.
    """
    if not isinstance(root_pid, int) or root_pid <= 1:
        raise ValueError("invalid_root_pid")
    table = _read_proc_table() if sys.platform.startswith("linux") else _read_ps_table()
    return build_snapshot(root_pid, table)


def _alive(pid: int) -> bool:
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)  # reaps our own zombie children
        if reaped == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    try:
        if sys.platform.startswith("linux"):
            text = Path(f"/proc/{pid}/stat").read_text(errors="replace")
            return text[text.rfind(")") + 1:].split()[0] == "Z"
        completed = subprocess.run([_PS_EXECUTABLE, "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                                   timeout=5, env={"PATH": os.defpath, "LC_ALL": "C"})
        return completed.stdout.strip().startswith("Z")
    except (OSError, subprocess.SubprocessError, IndexError):
        return False


def terminate_tree(snapshot: TreeSnapshot, *, grace_s: float = 3.0, verify_s: float = 2.0,
                   signal_groups: Iterable[int] | None = None) -> TerminationReport:
    """Terminate every process of ``snapshot``: SIGTERM their groups, wait, SIGKILL survivors, verify.

    Parameters
    ----------
    snapshot:
        From ``process_tree_snapshot``.  Empty snapshot -> trivially confirmed report.
    grace_s:
        Seconds (default 3.0, >= 0) to wait after SIGTERM before SIGKILL.
    verify_s:
        Seconds (default 2.0, >= 0) to wait after SIGKILL for the processes to disappear before they are
        reported as survivors.

    signal_groups:
        Process groups that may be signalled with ``killpg``; a member in any other group is signalled by pid.
        Default None: every member's group.

    Every distinct pgid of a member is signalled with ``killpg`` (bash-tool children sit in their own groups); a
    member in the caller's own process group, or with pgid <= 1, is signalled by pid instead so the caller is never
    killed.  The calling process itself is never signalled (listed in ``skipped_self``).  Own child zombies are
    reaped during verification.  Caveat: a pid recycled between snapshot and kill could be signalled; the window is
    the caller's snapshot-to-kill latency.
    """
    started = time.monotonic()
    own_pid, own_pgid = os.getpid(), os.getpgrp()
    members = [item for item in snapshot.members if item.pid != own_pid]
    skipped = tuple(item.pid for item in snapshot.members if item.pid == own_pid)
    allowed = None if signal_groups is None else set(signal_groups)
    groups = [g for g in dict.fromkeys(item.pgid for item in members)
              if g > 1 and g != own_pgid and (allowed is None or g in allowed)]
    loose = [item.pid for item in members if item.pgid not in groups]
    errors: list[str] = []

    def send(sig: int) -> None:
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"killpg:{group}:{errno_name(error)}")
        for pid in loose:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"kill:{pid}:{errno_name(error)}")

    def wait_gone(pids: list[int], seconds: float) -> list[int]:
        deadline = time.monotonic() + max(0.0, seconds)
        remaining = [pid for pid in pids if _alive(pid)]
        while remaining and time.monotonic() < deadline:
            time.sleep(0.02)
            remaining = [pid for pid in remaining if _alive(pid)]
        return remaining

    pids = [item.pid for item in members]
    send(signal.SIGTERM)
    after_term = wait_gone(pids, grace_s)
    exited = tuple(pid for pid in pids if pid not in after_term)
    survivors = after_term
    killed: tuple[int, ...] = ()
    if after_term:
        send(signal.SIGKILL)
        killed = tuple(after_term)
        survivors = wait_gone(after_term, verify_s)
    return TerminationReport(
        root_pid=snapshot.root_pid, members=tuple(pids), signalled_groups=tuple(groups), signalled_pids=tuple(loose),
        exited_after_term=exited, killed=killed, survivors=tuple(survivors), skipped_self=skipped,
        errors=tuple(errors), elapsed_s=time.monotonic() - started)


# ----------------------------------------------------------------------------------------- agent process marker

# Environment variable carrying the agent session's instance id into every process the agent starts.
AGENT_MARKER_ENV = "AURIGA_AGENT_INSTANCE"
_MARKER_VALUE = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


def agent_marker(instance_id: str) -> str:
    """``instance_id`` validated as a marker value (1-128 of ``A-Za-z0-9_.:-``); ``ValueError`` otherwise."""
    if not isinstance(instance_id, str) or not _MARKER_VALUE.match(instance_id):
        raise ValueError("invalid_agent_marker")
    return instance_id


@dataclass(frozen=True)
class MarkedProcess:
    """A process selected by ``processes_with_marker``: ``pid``/``ppid``/``pgid`` and why (``reason``): ``"env"``
    (its environment carries the marker), ``"cwd"`` (environment unreadable, cwd inside an agent directory, orphaned
    or child of a marked process) or ``"descendant"`` (a child of a marked process)."""
    pid: int
    ppid: int
    pgid: int
    reason: str

    def info(self) -> ProcessInfo:
        return ProcessInfo(self.pid, self.ppid, self.pgid)


@dataclass(frozen=True)
class MarkerScan:
    """Result of one ``processes_with_marker`` scan.

    ``found``: the marked processes. ``table``: every process row read (all users; used to decide which process
    groups consist only of selected processes). ``env_unreadable``: number of same-uid processes whose environment
    the platform withholds (macOS platform binaries), i.e. that only the cwd fallback can select.
    """
    instance_id: str | None
    found: tuple[MarkedProcess, ...]
    table: tuple[ProcessInfo, ...]
    env_unreadable: int = 0

    @property
    def pids(self) -> tuple[int, ...]:
        return tuple(item.pid for item in self.found)

    def to_dict(self) -> dict:
        return {"found": [{"pid": item.pid, "pgid": item.pgid, "reason": item.reason} for item in self.found],
                "env_unreadable": self.env_unreadable}


@dataclass(frozen=True)
class _Row:
    info: ProcessInfo
    uid: int
    env: tuple[bytes, ...] | None      # None: not readable (or empty)
    cwd: str | None


def _within(path: str | None, roots: Sequence[str]) -> bool:
    if not path:
        return False
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:
            continue
    return False


def select_marked(rows: Iterable, instance_id: str | None, *, cwd_roots: Sequence[str | os.PathLike] = (),
                  uid: int | None = None, own_pid: int | None = None) -> tuple[MarkedProcess, ...]:
    """Pure selection rule of ``processes_with_marker`` over process rows (see the module doc).

    rows: objects with ``info`` (``ProcessInfo``), ``uid``, ``env`` (tuple of ``b"KEY=VALUE"`` entries, or None when
    unreadable) and ``cwd`` (real path or None). instance_id: marker value or None (cwd fallback only). cwd_roots:
    agent directories (resolved here). uid / own_pid: default this process's. Never selects pid <= 1, ``own_pid`` or
    any ancestor of ``own_pid``; only rows of ``uid``. Returns marked processes ordered by pid.
    """
    uid = os.getuid() if uid is None else uid
    own_pid = os.getpid() if own_pid is None else own_pid
    roots = [os.path.realpath(os.fspath(root)) for root in cwd_roots]
    entry = f"{AGENT_MARKER_ENV}={agent_marker(instance_id)}".encode() if instance_id is not None else None
    rows = list(rows)
    by_pid = {row.info.pid: row for row in rows}
    excluded, cursor = {own_pid}, by_pid.get(own_pid)
    while cursor is not None and cursor.info.ppid not in excluded and cursor.info.ppid > 1:
        excluded.add(cursor.info.ppid)
        cursor = by_pid.get(cursor.info.ppid)
    mine = [row for row in rows if row.uid == uid and row.info.pid > 1 and row.info.pid not in excluded]
    marked: dict[int, str] = {}
    for row in mine:
        if entry is not None and row.env is not None and entry in row.env:
            marked[row.info.pid] = "env"
    changed = True
    while changed:
        changed = False
        for row in mine:
            if row.info.pid in marked:
                continue
            if row.info.ppid in marked:
                marked[row.info.pid] = "descendant"
                changed = True
            elif row.env is None and row.info.ppid == 1 and _within(row.cwd, roots):
                marked[row.info.pid] = "cwd"
                changed = True
    return tuple(MarkedProcess(pid, by_pid[pid].info.ppid, by_pid[pid].info.pgid, reason)
                 for pid, reason in sorted(marked.items()))


def _linux_rows(proc_root: str | os.PathLike = "/proc") -> list[_Row]:
    uid, rows = os.getuid(), []
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        base = Path(proc_root, entry)
        try:
            owner = os.stat(base).st_uid
            info = parse_proc_stat(int(entry), (base / "stat").read_text(errors="replace"))
        except OSError:
            continue  # exited while scanning
        if info is None:
            continue
        env = cwd = None
        if owner == uid:
            try:
                data = (base / "environ").read_bytes()
                env = tuple(item for item in data.split(b"\0") if item) or None
            except OSError:
                env = None
            try:
                cwd = os.readlink(base / "cwd")
            except OSError:
                cwd = None
        rows.append(_Row(info, owner, env, cwd))
    return rows


class _Darwin:
    """``sysctl(KERN_PROCARGS2)`` (environment) and ``proc_pidinfo(PROC_PIDVNODEPATHINFO)`` (cwd), loaded lazily."""
    _instance = None
    CTL_KERN, KERN_ARGMAX, KERN_PROCARGS2, PROC_PIDVNODEPATHINFO, MAXPATHLEN = 1, 8, 49, 9, 1024

    def __init__(self):
        import ctypes
        self.ctypes = ctypes
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        mib = (ctypes.c_int * 2)(self.CTL_KERN, self.KERN_ARGMAX)
        value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if self.libc.sysctl(mib, 2, ctypes.byref(value), ctypes.byref(size), None, 0) != 0 or value.value <= 0:
            raise OSError("kern_argmax_unavailable")
        self.argmax = value.value
        # struct proc_vnodepathinfo = 2 x vnode_info_path (vnode_info 152 bytes + MAXPATHLEN); cwd path at offset 152
        self.info_size, self.path_offset = 2 * (152 + self.MAXPATHLEN), 152

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def env(self, pid: int, buffer) -> tuple[bytes, ...] | None:
        ctypes = self.ctypes
        mib = (ctypes.c_int * 3)(self.CTL_KERN, self.KERN_PROCARGS2, pid)
        size = ctypes.c_size_t(len(buffer))
        if self.libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0 or size.value < 4:
            return None
        data = buffer.raw[:size.value]
        argc = int.from_bytes(data[:4], sys.byteorder)
        parts = data[4:].split(b"\0")
        index = 1                                   # parts[0] = executable path, then NUL padding
        while index < len(parts) and parts[index] == b"":
            index += 1
        index += argc                               # the arguments
        env = []
        while index < len(parts) and parts[index]:
            env.append(parts[index])
            index += 1
        return tuple(env) or None                   # platform binaries: the kernel withholds every entry

    def cwd(self, pid: int) -> str | None:
        ctypes = self.ctypes
        buffer = ctypes.create_string_buffer(self.info_size)
        if self.libproc.proc_pidinfo(pid, self.PROC_PIDVNODEPATHINFO, ctypes.c_uint64(0), buffer,
                                     self.info_size) != self.info_size:
            return None
        raw = buffer.raw[self.path_offset:self.path_offset + self.MAXPATHLEN]
        path = raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
        return path or None


def _darwin_rows() -> list[_Row]:
    completed = subprocess.run([_PS_EXECUTABLE, "-axo", "pid=,ppid=,pgid=,uid="], capture_output=True, text=True,
                               timeout=10, check=True, env={"PATH": os.defpath, "LC_ALL": "C"})
    darwin, uid, rows = _Darwin.get(), os.getuid(), []
    buffer = darwin.ctypes.create_string_buffer(darwin.argmax)
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or not all(item.lstrip("-").isdigit() for item in fields[:4]):
            continue
        pid, ppid, pgid, owner = (int(item) for item in fields[:4])
        env = cwd = None
        if owner == uid:
            env, cwd = darwin.env(pid, buffer), darwin.cwd(pid)
        rows.append(_Row(ProcessInfo(pid, ppid, pgid), owner, env, cwd))
    return rows


def _marker_rows() -> list[_Row]:
    if sys.platform.startswith("linux"):
        return _linux_rows()
    if sys.platform == "darwin":
        return _darwin_rows()
    raise OSError("agent_marker_scan_unsupported_platform")


def processes_with_marker(instance_id: str | None, *, cwd_roots: Sequence[str | os.PathLike] = ()) -> MarkerScan:
    """Find every same-uid process that belongs to the agent session ``instance_id`` (see the module doc).

    instance_id: the value of ``AGENT_MARKER_ENV`` in the agent environment (``clean_environment(instance_id=)``),
    or None to use only the cwd fallback. cwd_roots: agent directories (normally the actor dir, the agent's cwd and
    HOME) for processes whose environment the platform withholds. One process-table read: about 30 ms for 400
    processes on macOS. Never selects this process, its ancestors or pid <= 1. Raises ``OSError`` when the process
    table cannot be read (callers then report the teardown unconfirmed).
    """
    rows = _marker_rows()
    found = select_marked(rows, instance_id, cwd_roots=cwd_roots)
    uid = os.getuid()
    unreadable = sum(1 for row in rows if row.uid == uid and row.env is None)
    return MarkerScan(instance_id, found, tuple(row.info for row in rows), unreadable)


@dataclass
class AgentTeardownReport:
    """Result of ``terminate_agent_processes``: ``termination`` (first kill pass over tree + marked processes),
    ``rounds`` (all kill passes), ``marker`` (scan facts), ``remaining`` (marked pids still alive after the last
    re-scan), ``errors`` (scan failures). ``confirmed``: every targeted process is gone AND the final re-scan found no
    marked process AND every scan succeeded."""
    termination: TerminationReport
    rounds: list = field(default_factory=list)
    marker: dict = field(default_factory=dict)
    remaining: tuple[int, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def confirmed(self) -> bool:
        return (all(item.confirmed for item in self.rounds) and not self.remaining and not self.errors)

    def to_dict(self) -> dict:
        result = self.termination.to_dict()
        result.update(confirmed=self.confirmed, rounds=len(self.rounds), marker=dict(self.marker),
                      remaining=list(self.remaining), scan_errors=list(self.errors))
        return result


def terminate_agent_processes(snapshot: TreeSnapshot, instance_id: str | None, *,
                              cwd_roots: Sequence[str | os.PathLike] = (), grace_s: float = 3.0,
                              verify_s: float = 2.0, max_rounds: int = 3, scan=None) -> AgentTeardownReport:
    """Stop the agent's process tree and every process carrying its marker, then verify by re-scanning.

    snapshot: the agent's tree (``process_tree_snapshot`` taken while the root ran, re-validated by the caller).
    instance_id / cwd_roots: as ``processes_with_marker``. grace_s / verify_s: per kill pass, as ``terminate_tree``.
    max_rounds: kill passes at most (default 3): a marked process found by a re-scan (forked during the previous
    pass) is killed in the next. scan: ``(instance_id, cwd_roots=) -> MarkerScan`` replacement (tests).

    Process groups are signalled with ``killpg`` only when the group belongs to the agent: a group of a tree member or
    of an environment-marked process (or its descendant), or a group whose every process was selected; a process
    selected only by the cwd fallback in any other group is signalled by pid. This process and its group are never
    signalled. The receipt is honest: ``confirmed`` is False when a scan failed or a marked process remains.
    """
    scanner = scan or processes_with_marker
    errors: list[str] = []
    rounds: list[TerminationReport] = []
    marker_facts: dict = {"instance_id_set": instance_id is not None}
    members = {item.pid: item for item in snapshot.members}
    tree_groups = {item.pgid for item in snapshot.members}

    def scan_once():
        try:
            return scanner(instance_id, cwd_roots=cwd_roots)
        except Exception as error:  # noqa: BLE001 - an unreadable process table is an unverifiable teardown
            errors.append(f"marker_scan:{type(error).__name__}")
            return None

    def targets(found_scan, base):
        selected = dict(base)
        groups = set(tree_groups) if base is members else set()
        if found_scan is not None:
            for item in found_scan.found:
                selected.setdefault(item.pid, item.info())
                if item.reason != "cwd":
                    groups.add(item.pgid)
            chosen = set(selected)
            by_group: dict[int, list[int]] = {}
            for row in found_scan.table:
                by_group.setdefault(row.pgid, []).append(row.pid)
            for group in {item.pgid for item in selected.values()}:
                if all(pid in chosen for pid in by_group.get(group, ())):
                    groups.add(group)
        return selected, groups

    found = scan_once()
    if found is not None:
        marker_facts.update(found.to_dict())
    selected, groups = targets(found, members)
    termination = terminate_tree(TreeSnapshot(snapshot.root_pid, tuple(selected.values())), grace_s=grace_s,
                                 verify_s=verify_s, signal_groups=groups)
    rounds.append(termination)
    remaining: tuple[int, ...] = ()
    for _ in range(max(1, max_rounds)):
        again = scan_once()
        if again is None:
            break
        remaining = tuple(pid for pid in again.pids if _alive(pid))
        if not remaining or len(rounds) >= max_rounds:
            break
        retry, retry_groups = targets(again, {})
        rounds.append(terminate_tree(TreeSnapshot(snapshot.root_pid, tuple(retry.values())), grace_s=grace_s,
                                     verify_s=verify_s, signal_groups=retry_groups))
    return AgentTeardownReport(termination=termination, rounds=rounds, marker=marker_facts, remaining=remaining,
                               errors=tuple(errors))


def stop_agent_tree(root_pid: int | None, root_alive: bool, known: Iterable[ProcessInfo], instance_id: str | None, *,
                    cwd_roots: Sequence[str | os.PathLike] = (), grace_s: float = 3.0,
                    verify_s: float = 2.0) -> AgentTeardownReport:
    """The shared close-time teardown of an agent backend (Claude Code CLI, OpenCode server); one call.

    root_pid: the agent process (None when it was never observed); root_alive: whether it still runs; known: every
    member seen by the session's periodic tree refreshes; instance_id: the marker of the session's environment
    (``clean_environment(instance_id=)``); cwd_roots: the agent directories (actor dir). Reads a fresh process table
    right before the kill, re-validates the tree (``teardown_snapshot``) and hands it to
    ``terminate_agent_processes`` (tree plus every marked process, killed, re-scanned). The receipt's ``confirmed`` is
    the only teardown verdict a session reports.
    """
    snapshot = (teardown_snapshot(root_pid, root_alive, list(known), process_table()) if root_pid
                else TreeSnapshot(0, ()))
    return terminate_agent_processes(snapshot, instance_id, cwd_roots=cwd_roots, grace_s=grace_s, verify_s=verify_s)


def errno_name(error: OSError) -> str:
    import errno as _errno
    return _errno.errorcode.get(error.errno or 0, "UNKNOWN")
