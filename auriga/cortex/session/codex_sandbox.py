"""A small bubblewrap boundary for the native Codex actor process.

The Harness remains outside this process boundary. Only the actor workspace,
Codex home, native executable, selected source snapshot, and run-owned IPC are
mounted into the actor's filesystem namespace.

The backend-neutral mechanism (supervisor, loopback relays, PyPI proxy, bwrap
argv, runtime mounts) is ``agent_sandbox.AgentSandbox``; this module adds the
Codex specifics and keeps the Codex argv byte-identical to the pre-extraction
implementation (``tests/test_agent_sandbox.py`` replays a golden).
"""

from __future__ import annotations

import os
# Modules the sandbox code uses stay importable through this module path for
# callers and tests that patch them here (e.g. ``codex_sandbox.socket.getaddrinfo``).
import importlib.util  # noqa: F401
import http.client  # noqa: F401
import ipaddress  # noqa: F401
import re
from pathlib import Path
import shutil
import socket  # noqa: F401
import ssl  # noqa: F401
import stat
import subprocess  # noqa: F401
import threading  # noqa: F401

from .agent_sandbox import _SUPERVISOR, AgentSandbox, _UnixForwarder  # noqa: F401  (historical import path)


class CodexSandbox(AgentSandbox):
    """Build and own the isolated native actor command prefix.

    ``read_paths`` is a sequence of ``(host_path, sandbox_path)`` read-only
    mounts. The caller appends the native Codex argv to the returned prefix.

    Codex specifics on top of ``AgentSandbox``: ``CODEX_HOME`` (read-write at
    its host path), the native executable at its resolved host path,
    ``CODEX_CA_CERTIFICATE`` when ``<codex_home>/account-control-ca.pem``
    exists at prepare time, and the pinned ``/tmp/codex-daemon-<uid>`` socket
    directory (``<ipc>/daemon``) behind ``repair_socket_alias`` and
    ``host_socket_path``.
    """
    error_prefix = "codex_sandbox"

    def __init__(self, private_root, ipc, actor_dir, codex_home, executable, *, ports=(), packages=True,
                 read_paths=()):
        self.codex_home = Path(codex_home).absolute()
        self.executable = Path(executable).resolve(strict=True)
        super().__init__(private_root, ipc, actor_dir, state_dirs=[(self.codex_home, self.codex_home)],
                         executables=[(self.executable, self.executable)],
                         env={"CODEX_HOME": str(self.codex_home)}, ports=ports, packages=packages,
                         read_paths=read_paths)
        self.daemon_directory_fd = None
        self.native_socket_fd = None

    def _prepare_backend_ipc(self):
        daemon_dir = self.ipc / "daemon"
        self._safe_directory(daemon_dir)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.daemon_directory_fd = os.open(daemon_dir, directory_flags)

    def _backend_mounts(self):
        return [(self.ipc / "daemon", Path(f"/tmp/codex-daemon-{os.getuid()}"), False)]

    def _backend_environment(self):
        ca_path = self.codex_home / "account-control-ca.pem"
        return {"CODEX_CA_CERTIFICATE": str(ca_path)} if ca_path.is_file() else {}

    def close(self):
        super().close()
        for descriptor_name in ("native_socket_fd", "daemon_directory_fd"):
            descriptor = getattr(self, descriptor_name, None)
            if descriptor is not None:
                try: os.close(descriptor)
                except OSError: pass
                setattr(self, descriptor_name, None)
        daemon_dir = self.ipc / "daemon"
        if daemon_dir.is_dir() and not daemon_dir.is_symlink():
            shutil.rmtree(daemon_dir)

    def repair_socket_alias(self, socket_path):
        """Pin Codex's namespace-only socket inode and repair its shared alias.

        Codex binds its Unix listener below /tmp/codex-daemon-UID even when the
        requested listen URI is in the run IPC directory. Its directory FD is
        pinned before the actor starts, and O_PATH|O_NOFOLLOW pins the socket
        inode itself. Host clients use host_socket_path so actor changes to the
        writable IPC directory or alias cannot redirect a connection.
        """
        requested = Path(socket_path)
        daemon = self.ipc / "daemon"
        if requested.parent != self.ipc or requested.name != "codex.sock":
            raise ValueError("codex_sandbox_socket_alias_not_run_owned")
        try:
            target = Path(os.readlink(requested))
        except (OSError, ValueError):
            return False
        expected_parent = Path(f"/tmp/codex-daemon-{os.getuid()}")
        if target.parent != expected_parent or not re.fullmatch(r"[0-9a-f]{64}", target.name):
            raise ValueError("codex_sandbox_unexpected_native_socket_target")
        try:
            if self.daemon_directory_fd is None:
                raise RuntimeError("codex_sandbox_socket_directory_not_pinned")
            pinned_directory = os.fstat(self.daemon_directory_fd)
            current_directory = daemon.lstat()
            if (not stat.S_ISDIR(current_directory.st_mode)
                    or (current_directory.st_dev, current_directory.st_ino)
                    != (pinned_directory.st_dev, pinned_directory.st_ino)):
                raise ValueError("codex_sandbox_native_socket_directory_changed")
            flags = getattr(os, "O_PATH", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            if not flags:
                raise RuntimeError("codex_sandbox_socket_descriptor_pinning_unavailable")
            native_fd = os.open(target.name, flags, dir_fd=self.daemon_directory_fd)
            info = os.fstat(native_fd)
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                os.close(native_fd)
                return False
        except (FileNotFoundError, NotADirectoryError):
            return False
        previous_fd = self.native_socket_fd
        self.native_socket_fd = native_fd
        if previous_fd is not None:
            try: os.close(previous_fd)
            except OSError: pass
        actual = daemon / target.name
        replacement = self.ipc / ".codex.sock.repair"
        replacement.unlink(missing_ok=True)
        os.symlink(actual, replacement)
        os.replace(replacement, requested)
        return True

    @property
    def host_socket_path(self):
        """A host-only path to the pinned native socket inode."""
        if self.native_socket_fd is None:
            raise RuntimeError("codex_sandbox_native_socket_not_pinned")
        return Path(f"/proc/{os.getpid()}/fd/{self.native_socket_fd}")
