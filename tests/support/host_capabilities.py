"""Host capability probes for sandbox / OS-isolation tests.

Sandbox and OS-isolation tests run only on hosts that actually provide the kernel feature, system tool or library
they exercise; elsewhere they are skipped with an explicit reason instead of failing or hanging. Each probe checks
the real host capability (it runs the production lookup or a minimal kernel call), never the platform name alone,
and is cached for the life of the test process. Probes have no lasting side effects: anything that changes process
state (namespaces, seccomp filters) runs in a short-lived child process.

Usage in a test module (``tests`` is on ``sys.path``)::

    from support.host_capabilities import require, program_isolation

    @require(program_isolation, "bounded programs run under Landlock + seccomp")
    class ProgramIsolationTest(unittest.TestCase): ...

``require`` decorates a test method or a whole ``TestCase`` class. On a host without the capability the test is
reported as skipped with ``requires host capability <name>: <reason> [probe: <why it is missing>]``.

Strict mode: with ``AURIGA_REQUIRE_SANDBOX_TESTS=1`` a missing capability is a test failure instead of a skip (a
decorated method fails; a decorated class errors in ``setUpClass``). Set it on hosts that are supposed to support
the sandbox (Linux benchmark hosts) so a capable CI host can never silently skip the sandbox tests, whether the
probe fails because the host lost the capability or because the production code it reuses regressed.

Report: ``PYTHONPATH=tests python -m support.host_capabilities`` prints every probe result for this host.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import functools
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import textwrap
import unittest

REQUIRE_ENV = "AURIGA_REQUIRE_SANDBOX_TESTS"
# Native architectures whose Linux syscall numbers the production isolation code hard-codes
# (auriga/runtime/programs/isolation.py launch(), native_terminal_replay._isolate()).
SUPPORTED_ARCHITECTURES = ("x86_64", "aarch64")
_REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Capability:
    """One probe result. ``detail`` says what was found, or why the capability is missing."""
    name: str
    available: bool
    detail: str

    def __bool__(self):
        return self.available


class MissingHostCapability(AssertionError):
    """Raised instead of skipping when ``AURIGA_REQUIRE_SANDBOX_TESTS=1`` and a capability is missing."""


def required():
    """True when missing capabilities must fail tests (``AURIGA_REQUIRE_SANDBOX_TESTS=1``)."""
    return os.environ.get(REQUIRE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _probe(function):
    """Cache a probe and turn an unexpected probe exception into an explicit "missing" result."""
    @functools.wraps(function)
    @functools.lru_cache(maxsize=None)
    def wrapper():
        try:
            return function()
        except Exception as error:  # A crashing probe must never break test discovery.
            return Capability(function.__name__, False, f"probe error {type(error).__name__}: {error}")
    return wrapper


def _linux_architecture():
    """None when the production Linux syscall tables apply to this host, else why not."""
    if platform.system() != "Linux":
        return f"not a Linux kernel ({platform.system()})"
    if platform.machine() not in SUPPORTED_ARCHITECTURES:
        return f"unsupported architecture {platform.machine()} (production syscall table: x86_64, aarch64)"
    return None


def _child(source, timeout):
    """Run ``source`` in a fresh interpreter importing this checkout; (ok, detail)."""
    try:
        # ``-c`` puts the working directory (this checkout) first on sys.path, ahead of any installed copy.
        result = subprocess.run([sys.executable, "-c", source], cwd=_REPOSITORY, capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"probe child timed out after {timeout:g} s"
    except OSError as error:
        return False, f"probe child could not start: {error}"
    output = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode == 0, (output[-1] if output else f"probe child exit status {result.returncode}")


@_probe
def linux_proc():
    """Linux procfs process identity: ``/proc/<pid>/stat`` start ticks and ``/proc/<pid>/cmdline``.

    Checks: the production reader ``auriga.apps.evaluation.completion.process_identity`` returns this process's
    start ticks, and ``/proc/self/cmdline`` is readable (queue PID adoption, pidfd cleanup and completion hooks
    bind processes this way). macOS: missing (no procfs). Linux: available. No subprocess, no timeout.
    """
    from auriga.apps.evaluation.completion import process_identity
    try:
        ticks = process_identity(os.getpid())
    except (OSError, ValueError, IndexError) as error:
        return Capability("linux_proc", False, f"/proc/<pid>/stat unreadable: {error}")
    if type(ticks) is not int or ticks <= 0:
        return Capability("linux_proc", False, "no /proc/<pid>/stat start ticks (procfs absent)")
    try:
        Path("/proc/self/cmdline").read_bytes()
    except OSError as error:
        return Capability("linux_proc", False, f"/proc/self/cmdline unreadable: {error}")
    return Capability("linux_proc", True, f"procfs start ticks {ticks}")


@_probe
def linux_runtime_tools():
    """Linux runtime publication tools used by ``auriga/cortex/session/system_runtime.py``.

    Checks: ``ldd`` on ``os.defpath`` and ``ldconfig`` on ``os.defpath:/sbin:/usr/sbin`` (the exact production
    lookups), and that the production ``system_library_path`` resolves ``libc.so.6`` through ``ldconfig -p``
    (a glibc runtime the sandbox can publish). macOS: missing (no ldd/ldconfig). glibc Linux: available.
    Timeout: the production lookup's own 10 s ``ldconfig -p`` timeout.
    """
    if not shutil.which("ldd", path=os.defpath):
        return Capability("linux_runtime_tools", False, "ldd not found on os.defpath")
    if not shutil.which("ldconfig", path=os.defpath + ":/sbin:/usr/sbin"):
        return Capability("linux_runtime_tools", False, "ldconfig not found on os.defpath:/sbin:/usr/sbin")
    from auriga.cortex.session.system_runtime import system_library_path
    try:
        libc = system_library_path("libc.so.6")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return Capability("linux_runtime_tools", False, f"ldconfig cannot resolve libc.so.6: {error}")
    return Capability("linux_runtime_tools", True, f"ldd, ldconfig, {libc}")


def landlock_abi():
    """Landlock ABI version of this kernel (0 when unavailable); production version query, no restriction."""
    if _linux_architecture():
        return 0
    from auriga.runtime.programs.isolation import CREATE_RULESET
    # landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION) only reports the ABI; it creates nothing.
    abi = ctypes.CDLL(None, use_errno=True).syscall(CREATE_RULESET, 0, 0, 1)
    return abi if abi > 0 else 0


@_probe
def landlock():
    """Landlock filesystem restriction (``auriga.runtime.programs.isolation.restrict``).

    Checks: Linux on x86_64/aarch64 (the production syscall table) and the kernel reports a Landlock ABI >= 1
    through the same ``landlock_create_ruleset`` version query production uses (fails when the kernel lacks
    Landlock, it is disabled at boot, or a container seccomp profile blocks it). Nothing is restricted.
    macOS: missing. Linux >= 5.13 with Landlock enabled: available. In-process syscall, no timeout.
    """
    problem = _linux_architecture()
    if problem:
        return Capability("landlock", False, problem)
    abi = landlock_abi()
    if abi < 1:
        return Capability("landlock", False, f"kernel reports no Landlock ABI (errno {ctypes.get_errno()})")
    return Capability("landlock", True, f"Landlock ABI {abi}")


@_probe
def seccomp():
    """seccomp-BPF syscall filters through libseccomp (``isolation.restrict_program_syscalls``).

    Checks: Linux on x86_64/aarch64, then a child interpreter loads the production socket-denial filter
    (``restrict_program_syscalls(deny_socket_creation=True)``, which dlopens ``libseccomp.so.2``) and verifies
    that ``socket()`` is really denied with EPERM. The filter dies with the child. macOS: missing.
    Linux with libseccomp 2 and seccomp filters allowed: available. Timeout 20 s.
    """
    problem = _linux_architecture()
    if problem:
        return Capability("seccomp", False, problem)
    ok, detail = _child(textwrap.dedent("""
        import socket
        from auriga.runtime.programs.isolation import restrict_program_syscalls
        restrict_program_syscalls(deny_socket_creation=True)
        try:
            socket.socket()
        except PermissionError:
            print("socket creation denied by seccomp")
        else:
            raise SystemExit("seccomp filter loaded but socket creation was not denied")
    """), timeout=20)
    return Capability("seccomp", ok, detail)


@_probe
def program_isolation():
    """Bounded actor programs with the sandbox on (``ProgramRunner`` + ``auriga/runtime/programs/isolation.py``).

    Checks what the launcher and ``program_runtime(sandbox_enabled=True)`` need from the host: Landlock ABI >= 4
    (the runner always passes ``tcp_ports=[]``, which needs network rules), working seccomp filters, ``ldd`` and
    an ``ldconfig`` that resolves ``libseccomp.so.2`` (the production ``system_library_path`` lookup that
    publishes the library into the program's read set). macOS: missing. Linux >= 6.7 (Landlock ABI 4) with
    libseccomp: available. Timeout: 20 s seccomp child + 10 s ``ldconfig -p``.
    """
    for dependency in (landlock(), seccomp(), linux_runtime_tools()):
        if not dependency:
            return Capability("program_isolation", False, f"{dependency.name}: {dependency.detail}")
    abi = landlock_abi()
    if abi < 4:
        return Capability("program_isolation", False, f"Landlock ABI {abi} < 4 (program TCP rules need ABI 4)")
    from auriga.cortex.session.system_runtime import system_library_path
    try:
        library = system_library_path("libseccomp.so.2")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return Capability("program_isolation", False, f"ldconfig cannot resolve libseccomp.so.2: {error}")
    return Capability("program_isolation", True, f"Landlock ABI {abi}, seccomp, {library}")


@_probe
def native_replay_sandbox():
    """Sandboxed native terminal replay (``auriga/cortex/session/native_terminal_replay.py`` ``_isolate``).

    Checks what the replay helper needs before it execs the CLI: Landlock ABI >= 3 (write restriction including
    truncate), working seccomp filters, and libseccomp discoverable via ``ctypes.util.find_library("seccomp")``
    (the helper's own lookup). macOS: missing. Linux >= 6.2 with libseccomp: available. Timeout: 20 s
    seccomp child.
    """
    for dependency in (landlock(), seccomp()):
        if not dependency:
            return Capability("native_replay_sandbox", False, f"{dependency.name}: {dependency.detail}")
    abi = landlock_abi()
    if abi < 3:
        return Capability("native_replay_sandbox", False, f"Landlock ABI {abi} < 3 (replay needs ABI 3)")
    library = ctypes.util.find_library("seccomp")
    if not library:
        return Capability("native_replay_sandbox", False, "ctypes.util.find_library('seccomp') found nothing")
    return Capability("native_replay_sandbox", True, f"Landlock ABI {abi}, {library}")


@_probe
def bubblewrap_namespaces():
    """Bubblewrap namespace sandbox for the native Codex actor (``CodexSandbox.prepare``).

    Checks: ``bwrap`` and the system ``python3`` are found by the production lookups
    (``CodexSandbox._find_host_tool`` / ``_find_python_runtime``) and a namespace sandbox actually starts:
    first the production unprivileged user-namespace probe ``CodexSandbox._probe_user_namespace`` (runs python
    in a new user/pid/net/ipc/uts namespace with no capabilities); if that fails, the rootful fallback
    ``prepare`` also accepts (``sudo -n bwrap`` with ``setpriv``, probed with a minimal ``python -c pass``).
    macOS: missing (no bubblewrap). Linux with bubblewrap and unprivileged user namespaces (or passwordless
    sudo): available. Timeout: 5 s per bwrap attempt (production probe value) + 10 s runtime discovery.
    """
    from auriga.cortex.session.codex_sandbox import CodexSandbox
    from auriga.cortex.session.system_runtime import system_runtime_mounts
    bwrap = CodexSandbox._find_host_tool("bwrap")
    if not bwrap:
        return Capability("bubblewrap_namespaces", False, "bwrap not found on PATH or os.defpath")
    try:
        python = CodexSandbox._find_python_runtime()
    except RuntimeError as error:
        return Capability("bubblewrap_namespaces", False, str(error))
    if CodexSandbox._probe_user_namespace(bwrap, python, os.getuid(), os.getgid()):
        return Capability("bubblewrap_namespaces", True, f"{bwrap} with unprivileged user namespaces")
    sudo, setpriv = CodexSandbox._find_host_tool("sudo"), CodexSandbox._find_host_tool("setpriv")
    if not sudo or not setpriv:
        return Capability("bubblewrap_namespaces", False,
                          "unprivileged user-namespace probe failed and sudo/setpriv are unavailable")
    try:
        runtime_mounts = system_runtime_mounts()
        command = [sudo, "-n", bwrap, "--unshare-pid", "--unshare-net", "--unshare-ipc",
                   "--unshare-uts", "--dir", "/usr", "--symlink", "usr/bin", "/bin",
                   "--dir", "/lib", "--dir", "/lib64", "--proc", "/proc", "--dev", "/dev",
                   "--tmpfs", "/tmp"]
        parents = {parent for _, destination in runtime_mounts for parent in destination.parents
                   if parent != Path("/") and parent not in (Path("/usr"), Path("/lib"), Path("/lib64"))}
        for parent in sorted(parents, key=lambda path: (len(path.parts), str(path))):
            command.extend(["--dir", str(parent)])
        for source, destination in runtime_mounts:
            command.extend(["--ro-bind", str(source), str(destination)])
        command.extend(["--", str(python), "-c", "pass"])
        probe = subprocess.run(command, capture_output=True, timeout=5)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return Capability("bubblewrap_namespaces", False, f"rootful bwrap probe failed: {error}")
    if probe.returncode != 0:
        return Capability("bubblewrap_namespaces", False,
                          "unprivileged user-namespace probe and rootful sudo -n bwrap probe both failed")
    return Capability("bubblewrap_namespaces", True, f"{bwrap} via sudo -n (rootful fallback)")


@_probe
def bubblewrap_user_namespaces():
    """Bubblewrap with UNPRIVILEGED user namespaces only (the generic ``AgentSandbox`` path; no sudo fallback).

    Checks: ``bwrap`` and the system ``python3`` are found by the production lookups and the production probe
    ``AgentSandbox._probe_user_namespace`` starts python in a new user/pid/net/ipc/uts namespace. Hosts that only
    support the rootful ``sudo -n bwrap`` fallback report missing. macOS: missing. Timeout: 5 s (production value).
    """
    from auriga.cortex.session.agent_sandbox import AgentSandbox
    bwrap = AgentSandbox._find_host_tool("bwrap")
    if not bwrap:
        return Capability("bubblewrap_user_namespaces", False, "bwrap not found on PATH or os.defpath")
    try:
        python = AgentSandbox._find_python_runtime()
    except RuntimeError as error:
        return Capability("bubblewrap_user_namespaces", False, str(error))
    if AgentSandbox._probe_user_namespace(bwrap, python, os.getuid(), os.getgid()):
        return Capability("bubblewrap_user_namespaces", True, f"{bwrap} with unprivileged user namespaces")
    groups = os.getgroups()
    if set(groups) - {os.getgid()}:
        detail = (f"unprivileged user-namespace probe failed: supplementary groups visible in the user namespace "
                  f"(setgroups denied) - host groups {sorted(groups)} include more than the primary gid "
                  f"{os.getgid()}, so the kernel keeps them mapped to the overflow gid inside the namespace and the "
                  "probe's os.getgroups().issubset({gid}) assertion fails (production falls back to the rootful "
                  "sudo -n bwrap + setpriv --clear-groups path on such hosts; see bubblewrap_namespaces)")
    else:
        detail = "unprivileged user-namespace probe failed"
    return Capability("bubblewrap_user_namespaces", False, detail)


PROBES = (linux_proc, linux_runtime_tools, landlock, seccomp, program_isolation, native_replay_sandbox,
          bubblewrap_namespaces, bubblewrap_user_namespaces)


def require(capability, reason):
    """Decorator: run the test only on hosts where ``capability()`` (a probe above) is available.

    ``reason`` says what the test exercises that needs the capability. Without the capability the decorated
    method or ``TestCase`` class is skipped (``unittest.skip``) with the reason and the probe detail; with
    ``AURIGA_REQUIRE_SANDBOX_TESTS=1`` it fails with ``MissingHostCapability`` instead.
    """
    result = capability()
    if result.available:
        return lambda test: test
    message = f"requires host capability {result.name}: {reason} [probe: {result.detail}]"
    if not required():
        return unittest.skip(message)
    message += f" ({REQUIRE_ENV}=1 forbids skipping)"

    def fail(test):
        if isinstance(test, type):
            def setUpClass(cls):
                raise MissingHostCapability(message)
            test.setUpClass = classmethod(setUpClass)
            return test

        @functools.wraps(test)
        def failing(*args, **kwargs):
            raise MissingHostCapability(message)
        return failing
    return fail


if __name__ == "__main__":
    for probe in PROBES:
        outcome = probe()
        print(f"{outcome.name:24} {'available' if outcome.available else 'missing  '}  {outcome.detail}")
