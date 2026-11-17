"""Pure planning capture of the bubblewrap agent sandbox (WP R4-5).

``planning_host()`` stubs every host probe ``prepare()`` performs (host tools, system
runtime and shared-library discovery, the user-namespace probe, ``/etc/ssl/certs``, the
host pip package, uid/gid), so ``prepare()`` only plans: it creates the run-owned
directories and host-side Unix relays and returns the bwrap command prefix and
environment, without running bubblewrap. It works on macOS and Linux alike.

``capture_codex()`` records the Codex command prefix, environment (in order), relays,
library-dependency inputs and post-close IPC state for representative configurations.
The golden fixture was captured from the pre-extraction source (7d5c83b)::

    git archive 7d5c83b auriga | tar -x -C /tmp/r4-5-orig
    PYTHONPATH=/tmp/r4-5-orig mamba run -n Auriga python tests/agent_sandbox_golden_fixture.py --write 7d5c83b

and ``tests/test_agent_sandbox.py`` replays it against the current tree. The temporary
root is normalised to ``<ROOT>``.
"""
from contextlib import ExitStack, contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "codex_sandbox_golden.json"
UID, GID = 4242, 4343
# A representative published runtime: a shell alias, a stdlib directory and a loader alias below /lib.
SYSTEM_RUNTIME = (
    (Path("/usr/bin/dash"), Path("/usr/bin/sh")),
    (Path("/usr/bin/python3.12"), Path("/usr/bin/python3")),
    (Path("/usr/lib/python3.12/json"), Path("/usr/lib/python3.12/json")),
    (Path("/usr/lib/x86_64-linux-gnu/libc.so.6"), Path("/lib/x86_64-linux-gnu/libc.so.6")),
)
CODEX_SCENARIOS = {
    # Main actor (CodexSession._prepare): model + account relays, packages, /harness, account CA present.
    "actor_user_namespace": dict(user_namespace=True, rg=True, ports=[43002, 43001], packages=True,
                                 harness=True, ca=True),
    # sudo + setpriv fallback; a relay on the default package-proxy port moves the proxy to 18889.
    "actor_setpriv_fallback": dict(user_namespace=False, rg=True, ports=[18888, 43001], packages=True,
                                   harness=True, ca=False),
    # Evidence agent (codex_evidence_agent): no packages, so no rg, PIP_* or pip bootstrap.
    "evidence_no_packages": dict(user_namespace=True, rg=True, ports=[43001], packages=False,
                                 harness=False, ca=True),
}


def _executable(path, text="#!/bin/sh\nexit 0\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


@contextmanager
def planning_host(root, sandbox_class, *, user_namespace=True, rg=True):
    """Stub host probes for ``sandbox_class.prepare()``; yields the fake tool paths and recorded inputs."""
    root = Path(root)
    host = root / "host"
    names = ("bwrap", "sudo", "setpriv", "env", "python3") + (("rg",) if rg else ())
    tools = {name: str(_executable(host / "bin" / name)) for name in names}
    pip_package = host / "pip-origin" / "pip"
    pip_package.mkdir(parents=True, exist_ok=True)
    (pip_package / "__init__.py").write_text("__version__ = 'fixture'\n")
    # The module whose globals prepare() reads (codex_sandbox before the extraction, agent_sandbox after).
    module = sys.modules[sandbox_class.prepare.__module__]
    library_calls = []

    def which(name, mode=os.F_OK | os.X_OK, path=None):
        return tools.get(name)

    def runtime_library_mounts(objects):
        objects = [Path(item) for item in objects]
        library_calls.append([str(item) for item in objects])
        return tuple((host / "lib" / f"lib{item.name}.so.1", Path("/usr/lib/x86_64-linux-gnu") / f"lib{item.name}.so.1")
                     for item in objects)

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "pip":
            return SimpleNamespace(origin=str(pip_package / "__init__.py"))
        return real_find_spec(name, *args, **kwargs)

    real_is_dir = Path.is_dir

    def is_dir(self, *args, **kwargs):
        return True if str(self) == "/etc/ssl/certs" else real_is_dir(self, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(shutil, "which", which))
        stack.enter_context(mock.patch.object(module, "system_runtime_mounts", lambda: SYSTEM_RUNTIME))
        stack.enter_context(mock.patch.object(module, "runtime_library_mounts", runtime_library_mounts))
        stack.enter_context(mock.patch.object(sandbox_class, "_probe_user_namespace",
                                              staticmethod(lambda *args, **kwargs: user_namespace)))
        stack.enter_context(mock.patch.object(importlib.util, "find_spec", find_spec))
        stack.enter_context(mock.patch.object(Path, "is_dir", is_dir))
        stack.enter_context(mock.patch.object(os, "getuid", return_value=UID))
        stack.enter_context(mock.patch.object(os, "getgid", return_value=GID))
        yield SimpleNamespace(tools=tools, library_calls=library_calls, host=host)


def _root_independent_prefix(prefix, root):
    """Drop ``--dir`` entries for strict ancestors of the temporary root (they depend on the host's /tmp)."""
    ancestors = {str(path) for path in root.parents}
    result, index = [], 0
    while index < len(prefix):
        if prefix[index] == "--dir" and index + 1 < len(prefix) and prefix[index + 1] in ancestors:
            index += 2
            continue
        result.append(prefix[index])
        index += 1
    return result


def normalise(value, root):
    if isinstance(value, str):
        return value.replace(str(root), "<ROOT>")
    if isinstance(value, (list, tuple)):
        return [normalise(item, root) for item in value]
    if isinstance(value, dict):
        return {normalise(key, root): normalise(item, root) for key, item in value.items()}
    return value


def _ipc_listing(ipc):
    return sorted(path.relative_to(ipc).as_posix() for path in ipc.rglob("*")) if ipc.exists() else None


def capture_codex(scenario):
    """Plan one Codex sandbox scenario and return its normalised record."""
    from auriga.cortex.session.codex_sandbox import CodexSandbox
    with tempfile.TemporaryDirectory(prefix="r45-", dir="/tmp") as temporary:
        # Depth >= 4 on every host: <ROOT> parents then sort after every fixed parent directory (depth <= 3),
        # so the --dir order does not depend on where the host's /tmp is (/tmp vs /private/tmp).
        root = Path(temporary).resolve() / "golden" / "root"
        assert len(root.parts) - 1 >= 4 and max(len(target.parent.parts) - 1 for _, target in SYSTEM_RUNTIME) <= 3
        actor, ipc, private = root / "run/actor", root / "ipc", root / "run/private"
        codex_home, snapshot = private / "codex-home", private / "actor-source"
        for path in (actor, ipc, codex_home, snapshot / "auriga"):
            path.mkdir(parents=True, mode=0o700)
        (snapshot / "auriga/__init__.py").write_text("")
        if scenario["ca"]:
            (codex_home / "account-control-ca.pem").write_text("fixture-ca")
        executable = _executable(root / "host/codex-bin/codex")
        with planning_host(root, CodexSandbox, user_namespace=scenario["user_namespace"],
                           rg=scenario["rg"]) as host:
            sandbox = CodexSandbox(private / "sandbox", ipc, actor, codex_home, executable,
                                   ports=scenario["ports"], packages=scenario["packages"],
                                   read_paths=[(snapshot, Path("/harness"))] if scenario["harness"] else ())
            try:
                prefix, env = sandbox.prepare()
                record = dict(
                    command_prefix=_root_independent_prefix(list(prefix), root), env=[[key, value] for key, value in env.items()],
                    package_proxy_port=sandbox.package_proxy_port,
                    forwarders=[[str(item.path), item.port, item.packages, item.package_port]
                                for item in sandbox.forwarders],
                    library_objects=host.library_calls, ipc_prepared=_ipc_listing(ipc))
            finally:
                sandbox.close()
            record["ipc_after_close"] = _ipc_listing(ipc)
        return normalise(record, root)


def capture_all():
    return {name: capture_codex(scenario) for name, scenario in CODEX_SCENARIOS.items()}


def write_fixture(revision):
    from auriga.cortex.session import codex_sandbox
    print("capturing from", codex_sandbox.__file__)
    data = {"captured_from": {"revision": revision}, "scenarios": capture_all()}
    FIXTURE.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        raise SystemExit("usage: PYTHONPATH=<source tree> python tests/agent_sandbox_golden_fixture.py --write <revision>")
    write_fixture(sys.argv[2])
