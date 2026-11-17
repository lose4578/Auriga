import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from auriga.cortex.agents import isolation as iso

HOST_SECRETS = {"ANTHROPIC_API_KEY": "sk-host", "OPENAI_API_KEY": "sk-o", "AWS_SECRET_ACCESS_KEY": "x",
                "GITHUB_TOKEN": "t", "HTTPS_PROXY": "http://p", "SSH_AUTH_SOCK": "/s", "MY_PASSWORD": "pw",
                "LANG": "en_US.UTF-8", "TZ": "UTC", "PATH": "/evil", "HOME": "/evil"}


class EnvironmentTests(unittest.TestCase):
    def test_no_host_credentials_and_explicit_keys(self):
        env = iso.clean_environment(HOST_SECRETS, path_dirs=["/priv/bin"], home="/run/actor", tmpdir="/run/tmp",
                                    extra={"ANTHROPIC_API_KEY": "auriga-local-relay"})
        self.assertEqual(env["ANTHROPIC_API_KEY"], "auriga-local-relay")  # extra is the only source
        for key in ("OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "HTTPS_PROXY", "SSH_AUTH_SOCK",
                    "MY_PASSWORD"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PATH"], "/priv/bin:" + os.defpath)
        self.assertEqual((env["HOME"], env["TMPDIR"], env["LANG"], env["TZ"]),
                         ("/run/actor", "/run/tmp", "en_US.UTF-8", "UTC"))

    def test_ignores_os_environ(self):
        with mock.patch.dict(os.environ, {"AURIGA_PROBE_SECRET": "leak"}):
            env = iso.clean_environment({}, home="/h", tmpdir="/t")
        self.assertEqual(set(env), {"PATH", "HOME", "TMPDIR", "LANG"})
        self.assertEqual(env["LANG"], "C.UTF-8")

    def test_validation(self):
        with self.assertRaises(ValueError):
            iso.clean_environment({"BAD-NAME": "x"}, home="/h", tmpdir="/t")
        with self.assertRaises(ValueError):
            iso.clean_environment({}, home="relative", tmpdir="/t")
        with self.assertRaises(ValueError):
            iso.clean_environment({}, home="/h", tmpdir="/t", extra={"A": "x\0y"})


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["/bin/rm", "-rf", str(self.dir)]))
        self.probe = self.dir / "probe.py"
        self.probe.write_text("import json,os,sys\nprint(json.dumps({'env':dict(os.environ),'argv':sys.argv[1:]}))\n")
        self.probe_sh = self.dir / "probe"
        self.probe_sh.write_text(f"#!/bin/sh\nexec {sys.executable} {self.probe} \"$@\"\n")
        self.probe_sh.chmod(0o700)

    def run_launcher(self, env, reexport=(), host=None, args=()):
        launcher = iso.write_env_launcher(self.dir / "launcher", self.probe_sh, env, reexport=reexport)
        self.assertEqual(stat.S_IMODE(launcher.stat().st_mode), 0o700)
        host_env = {"PATH": os.defpath, **(host or {})}
        out = subprocess.run([str(launcher), *args], env=host_env, capture_output=True, text=True, check=True).stdout
        import json
        return json.loads(out)

    def test_only_given_env_and_reexports_visible(self):
        env = {"PATH": "/usr/bin:/bin", "HOME": "/run/actor", "CLAUDE_CODE_X": "1"}
        host = {"ANTHROPIC_API_KEY": "sk-host", "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                "CLAUDE_AGENT_SDK_VERSION": "0.2.180", "UNRELATED": "u"}
        result = self.run_launcher(env, reexport=["CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION",
                                                   "CLAUDE_CODE_SDK_READS_SESSION_STATE"], host=host,
                                   args=["--flag", "a b"])
        seen = {k: v for k, v in result["env"].items() if k not in ("LC_CTYPE", "__CF_USER_TEXT_ENCODING", "PWD", "SHLVL", "_")}
        self.assertEqual(seen, {**env, "CLAUDE_CODE_ENTRYPOINT": "sdk-py", "CLAUDE_AGENT_SDK_VERSION": "0.2.180"})
        self.assertEqual(result["argv"], ["--flag", "a b"])

    def test_awkward_values_roundtrip(self):
        values = {"A": "it's \"q\" $HOME `id` \\ *", "B": "line1\nline2", "C": "", "D": "  spaced  ", "E": "a=b=c",
                  "F": "$(touch /tmp/auriga-should-not-exist)"}
        result = self.run_launcher(values)
        for key, value in values.items():
            self.assertEqual(result["env"][key], value)

    def test_empty_reexport_value_is_preserved_and_unset_is_absent(self):
        result = self.run_launcher({"A": "1"}, reexport=["EMPTY_ONE", "MISSING_ONE"], host={"EMPTY_ONE": ""})
        self.assertEqual(result["env"]["EMPTY_ONE"], "")
        self.assertNotIn("MISSING_ONE", result["env"])

    def test_rejections(self):
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "l", "relative/cli", {})
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "l", "/bin/true", {"A": "1"}, reexport=["A"])
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "l", "/bin/true", {"A": "1"}, reexport=["bad name"])
        target = self.dir / "real"
        target.write_text("x")
        (self.dir / "link").symlink_to(target)
        with self.assertRaises(ValueError):
            iso.write_env_launcher(self.dir / "link", "/bin/true", {})
        self.assertEqual(target.read_text(), "x")


class PrivateDirTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["/bin/rm", "-rf", str(self.dir)]))

    def test_modes_nested_and_idempotent(self):
        old = os.umask(0o022)
        try:
            made = iso.private_dirs(self.dir, ["tmp", "opencode/xdg"])
            again = iso.private_dirs(self.dir, ["tmp"])
        finally:
            os.umask(old)
        self.assertEqual(made["tmp"], again["tmp"])
        for path in (made["tmp"], made["opencode/xdg"], made["opencode/xdg"].parent):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)

    def test_loose_existing_dir_is_tightened(self):
        (self.dir / "tmp").mkdir(mode=0o755)
        os.chmod(self.dir / "tmp", 0o755)
        iso.private_dirs(self.dir, ["tmp"])
        self.assertEqual(stat.S_IMODE((self.dir / "tmp").stat().st_mode), 0o700)

    def test_symlink_and_bad_names_refused(self):
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: subprocess.run(["/bin/rm", "-rf", str(outside)]))
        (self.dir / "tmp").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "private_dir_symlink"):
            iso.private_dirs(self.dir, ["tmp"])
        (self.dir / "mid").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "private_dir_symlink"):
            iso.private_dirs(self.dir, ["mid/child"])
        self.assertEqual(list(outside.iterdir()), [])
        for bad in ("../x", "/abs", "", "a/../b"):
            with self.assertRaises(ValueError):
                iso.private_dirs(self.dir, [bad])
        (self.dir / "file").write_text("x")
        with self.assertRaisesRegex(ValueError, "private_dir_not_directory"):
            iso.private_dirs(self.dir, ["file"])


class PinningTests(unittest.TestCase):
    def test_sha_cache_and_invalidation(self):
        import hashlib
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "cli"
            path.write_bytes(b"one")
            with mock.patch.object(iso, "_hash_file", wraps=iso._hash_file) as spy:
                self.assertEqual(iso.file_sha256_cached(path), hashlib.sha256(b"one").hexdigest())
                iso.file_sha256_cached(path)
                self.assertEqual(spy.call_count, 1)
                stamp = path.stat().st_mtime_ns
                path.write_bytes(b"two")  # same size
                os.utime(path, ns=(stamp + 5_000_000_000, stamp + 5_000_000_000))
                self.assertEqual(iso.file_sha256_cached(path), hashlib.sha256(b"two").hexdigest())
                self.assertEqual(spy.call_count, 2)
            link = Path(raw) / "link"
            link.symlink_to(path)
            self.assertEqual(iso.file_sha256_cached(link), hashlib.sha256(b"two").hexdigest())
            with self.assertRaises(FileNotFoundError):
                iso.file_sha256_cached(Path(raw) / "missing")

    def test_version_mismatch(self):
        self.assertEqual(iso.require_exact_version("claude_code", " 2.1.321\n", "2.1.321"), "2.1.321")
        for actual in ("2.1.322", "v2.1.321", "2.1.321 (Claude Code)", None, ""):
            with self.assertRaises(iso.VersionMismatch) as caught:
                iso.require_exact_version("opencode", actual, "2.1.321")
            self.assertEqual(str(caught.exception), "opencode_version_mismatch")
            self.assertEqual(caught.exception.pinned, "2.1.321")


PROC_STAT_FAKE = {
    100: "100 (agent) S 1 100 100 0 -1 4194560 1 0 0",
    101: "101 (bash tool) S 100 101 101 0 -1 0",          # detached: own group and session
    102: "102 (sleep)) weird ) S 101 101 101 0 -1 0",    # command contains ')' and spaces
    103: "103 (node) S 100 100 100 0 -1 0",
    200: "200 (other) S 1 200 200 0 -1 0",
    201: "201 (child of other) S 200 200 200 0 -1 0",
}


class SnapshotParserTests(unittest.TestCase):
    def test_proc_stat_parser(self):
        self.assertEqual(iso.parse_proc_stat(102, PROC_STAT_FAKE[102]), iso.ProcessInfo(102, 101, 101))
        self.assertIsNone(iso.parse_proc_stat(1, "garbage"))
        self.assertIsNone(iso.parse_proc_stat(1, "1 (x) S"))

    def test_ps_parser(self):
        rows = iso.parse_ps_table("  1     0     1\n 50  1  50\nbad line\n 7 x 7\n")
        self.assertEqual(rows, [iso.ProcessInfo(1, 0, 1), iso.ProcessInfo(50, 1, 50)])

    def test_linux_proc_fake_tree(self):
        with tempfile.TemporaryDirectory() as raw:
            for pid, text in PROC_STAT_FAKE.items():
                (Path(raw) / str(pid)).mkdir()
                (Path(raw) / str(pid) / "stat").write_text(text)
            (Path(raw) / "self").mkdir()
            (Path(raw) / "cpuinfo").write_text("x")
            (Path(raw) / "999").mkdir()  # exited: no stat file
            table = iso._read_proc_table(raw)
        snapshot = iso.build_snapshot(100, table)
        self.assertEqual([m.pid for m in snapshot.members], [100, 101, 103, 102])
        self.assertEqual(snapshot.pgids, (100, 101))
        self.assertEqual(iso.build_snapshot(555, table).members, ())

    def test_cycle_safe_and_invalid_root(self):
        table = [iso.ProcessInfo(5, 6, 5), iso.ProcessInfo(6, 5, 5)]
        self.assertEqual(len(iso.build_snapshot(5, table).members), 2)
        with self.assertRaises(ValueError):
            iso.process_tree_snapshot(1)


class TerminateTreeTests(unittest.TestCase):
    def spawn_tree(self):
        script = textwrap.dedent("""
            import os, subprocess, sys, time
            detached = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
            grouped = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            print(detached.pid, grouped.pid, flush=True)
            time.sleep(120)
        """)
        root = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                                start_new_session=True)
        detached, grouped = (int(x) for x in root.stdout.readline().split())
        self.addCleanup(self.reap, root, detached, grouped)
        return root, detached, grouped

    @staticmethod
    def reap(root, *pids):
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        root.kill()
        root.wait()
        root.stdout.close()

    def test_detached_child_found_and_killed(self):
        root, detached, grouped = self.spawn_tree()
        self.assertNotEqual(os.getpgid(detached), os.getpgid(root.pid))
        snapshot = iso.process_tree_snapshot(root.pid)
        self.assertEqual({m.pid for m in snapshot.members}, {root.pid, detached, grouped})
        # a plain killpg of the root group (what the old teardown did) leaves the detached child alive
        os.killpg(os.getpgid(root.pid), signal.SIGKILL)
        root.wait()
        time.sleep(0.2)
        self.assertTrue(iso._alive(detached))
        report = iso.terminate_tree(snapshot, grace_s=1.0)
        self.assertTrue(report.confirmed, report)
        self.assertEqual(report.errors, ())
        self.assertFalse(iso._alive(detached))
        self.assertFalse(iso._alive(grouped))
        self.assertIn(snapshot.members[0].pgid, report.signalled_groups)
        self.assertEqual(report.to_dict()["confirmed"], True)

    def test_sigterm_ignoring_process_needs_sigkill(self):
        script = ("import signal,time,sys\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                  "print('ready', flush=True)\ntime.sleep(120)\n")
        proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                                start_new_session=True)
        self.addCleanup(self.reap, proc)
        proc.stdout.readline()
        report = iso.terminate_tree(iso.process_tree_snapshot(proc.pid), grace_s=0.2)
        self.assertTrue(report.confirmed)
        self.assertEqual(report.killed, (proc.pid,))
        self.assertEqual(report.exited_after_term, ())

    def test_never_signals_caller_and_empty_snapshot(self):
        own = iso.TreeSnapshot(os.getpid(), (iso.ProcessInfo(os.getpid(), 1, os.getpgrp()),))
        report = iso.terminate_tree(own, grace_s=0)
        self.assertEqual(report.skipped_self, (os.getpid(),))
        self.assertEqual(report.signalled_groups, ())
        self.assertTrue(report.confirmed)
        empty = iso.terminate_tree(iso.TreeSnapshot(4242424, ()), grace_s=0)
        self.assertTrue(empty.confirmed)
        self.assertEqual(empty.members, ())

    def test_survivor_reported_unconfirmed(self):
        snapshot = iso.TreeSnapshot(77, (iso.ProcessInfo(77, 1, 77),))
        with mock.patch.object(iso, "_alive", return_value=True), mock.patch.object(iso.os, "killpg") as killpg:
            report = iso.terminate_tree(snapshot, grace_s=0, verify_s=0)
        self.assertFalse(report.confirmed)
        self.assertEqual(report.survivors, (77,))
        self.assertEqual([c.args for c in killpg.call_args_list], [(77, signal.SIGTERM), (77, signal.SIGKILL)])


class _Row:
    def __init__(self, pid, ppid, pgid, *, uid=None, env=None, cwd=None):
        self.info, self.uid, self.env, self.cwd = iso.ProcessInfo(pid, ppid, pgid), os.getuid() if uid is None else uid, \
            env, cwd


class MarkerSelectionTests(unittest.TestCase):
    """``select_marked``: the pure rule behind ``processes_with_marker`` (no real processes)."""

    MARK = (b"PATH=/usr/bin", iso.AGENT_MARKER_ENV.encode() + b"=agent-abc")

    def select(self, rows, **kwargs):
        kwargs.setdefault("cwd_roots", ["/runs/r1/actor"])
        kwargs.setdefault("own_pid", 50)
        with mock.patch.object(iso.os.path, "realpath", side_effect=lambda path: path):
            return {item.pid: item.reason for item in iso.select_marked(rows, "agent-abc", **kwargs)}

    def test_environment_marker_and_descendants(self):
        rows = [_Row(50, 40, 40), _Row(40, 1, 40),                                      # the gateway and its parent
                _Row(100, 50, 40, env=self.MARK),                                         # the CLI
                _Row(101, 1, 101, env=self.MARK),                                         # detached, env readable
                _Row(102, 101, 101),                                                      # its child, env unreadable
                _Row(103, 1, 103, env=(iso.AGENT_MARKER_ENV.encode() + b"=agent-other",)),  # another session
                _Row(104, 1, 104, env=self.MARK, uid=os.getuid() + 1)]                    # another user
        self.assertEqual(self.select(rows), {100: "env", 101: "env", 102: "descendant"})

    def test_cwd_fallback_only_for_unreadable_orphans_inside_the_agent_dirs(self):
        rows = [_Row(50, 1, 50),
                _Row(200, 1, 200, cwd="/runs/r1/actor"),               # macOS platform binary, orphaned: selected
                _Row(201, 1, 201, cwd="/runs/r1/actor/sub"),           # in a subdirectory: selected
                _Row(202, 300, 202, cwd="/runs/r1/actor"),             # an operator's shell (parent is a login)
                _Row(300, 1, 300, cwd="/"),
                _Row(203, 1, 203, cwd="/runs/r1/actor-old"),           # prefix only, not inside
                _Row(204, 1, 204, cwd="/runs/r1/actor", env=(b"PATH=/x",)),  # readable env without the marker
                _Row(205, 200, 200, cwd="/tmp")]                       # child of a selected orphan
        self.assertEqual(self.select(rows), {200: "cwd", 201: "cwd", 205: "descendant"})
        self.assertEqual(self.select(rows, cwd_roots=[]), {})

    def test_never_the_caller_its_ancestors_or_pid_one(self):
        rows = [_Row(1, 0, 1, env=self.MARK), _Row(30, 1, 30, env=self.MARK), _Row(50, 30, 30, env=self.MARK),
                _Row(60, 50, 30, env=self.MARK)]
        self.assertEqual(self.select(rows), {60: "env"})

    def test_marker_value_is_validated(self):
        for bad in ("", "a b", "x" * 129, None.__class__.__name__ + "\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                iso.agent_marker(bad)
        env = iso.clean_environment({}, home="/h", tmpdir="/t", instance_id="agent-1f")
        self.assertEqual(env[iso.AGENT_MARKER_ENV], "agent-1f")
        self.assertNotIn(iso.AGENT_MARKER_ENV, iso.clean_environment({}, home="/h", tmpdir="/t"))

    def test_linux_proc_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor"
            actor.mkdir()
            for pid, stat_line, environ in ((11, "11 (perl) S 1 11 11", b"PATH=/b\0" + self.MARK[1] + b"\0"),
                                            (12, "12 (sh) S 1 12 12", None)):
                base = root / "proc" / str(pid)
                base.mkdir(parents=True)
                (base / "stat").write_text(stat_line + " 0 0 0")
                if environ is not None:
                    (base / "environ").write_bytes(environ)
                (base / "cwd").symlink_to(actor)
            rows = {row.info.pid: row for row in iso._linux_rows(root / "proc")}
        self.assertEqual(rows[11].info, iso.ProcessInfo(11, 1, 11))
        self.assertIn(self.MARK[1], rows[11].env)
        self.assertIsNone(rows[12].env)                      # unreadable environ
        self.assertEqual(rows[12].cwd, str(actor))


class MarkerTeardownTests(unittest.TestCase):
    """``processes_with_marker`` / ``terminate_agent_processes`` against real processes of this host."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.actor = Path(os.path.realpath(directory.name)) / "actor"
        self.actor.mkdir()
        self.instance = "agent-test" + os.urandom(6).hex()
        self.env = iso.clean_environment({}, home=self.actor, tmpdir=self.actor, instance_id=self.instance)

    def kill_later(self, *pids):
        for pid in pids:
            self.addCleanup(lambda pid=pid: os.kill(pid, signal.SIGKILL) if iso._alive(pid) else None)

    def leave_detached_children(self):
        """A shell that leaves three children behind and exits at once (what a foreground Bash tool call does):
        ``sleep &`` (an Apple platform binary on macOS: environment withheld), a setsid python and a plain python."""
        script = ("sleep 300 >/dev/null 2>&1 & echo $!; "
                  f"{sys.executable} -c 'import os, time; os.setsid(); time.sleep(300)' >/dev/null 2>&1 & echo $!; "
                  f"{sys.executable} -c 'import time; time.sleep(300)' >/dev/null 2>&1 & echo $!")
        shell = subprocess.run(["/bin/sh", "-c", script], env=self.env, cwd=self.actor, capture_output=True, text=True,
                               timeout=10, start_new_session=True)
        pids = [int(line) for line in shell.stdout.split()]
        self.kill_later(*pids)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not all(self.parent(pid) == 1 for pid in pids):
            time.sleep(0.02)
        return pids

    @staticmethod
    def parent(pid):
        rows = {row.pid: row for row in (iso._read_proc_table() if sys.platform.startswith("linux")
                                         else iso._read_ps_table())}
        return rows[pid].ppid if pid in rows else None

    def test_detached_children_are_found_killed_and_verified(self):
        pids = self.leave_detached_children()
        self.assertTrue(all(iso._alive(pid) for pid in pids))
        scan = iso.processes_with_marker(self.instance, cwd_roots=[self.actor])
        self.assertTrue(set(pids) <= set(scan.pids), (pids, scan.to_dict()))
        self.assertNotIn(os.getpid(), scan.pids)
        report = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance, cwd_roots=[self.actor],
                                               grace_s=2.0)
        self.assertTrue(report.confirmed, report.to_dict())
        self.assertFalse(any(iso._alive(pid) for pid in pids))
        self.assertEqual(report.remaining, ())
        self.assertNotIn(os.getpgrp(), report.termination.signalled_groups)
        self.assertEqual(iso.processes_with_marker(self.instance, cwd_roots=[self.actor]).found, ())

    def test_unmarked_processes_are_left_alone(self):
        # A child of this process in the actor dir (an operator's shell is never an orphan) and a process of another
        # session's marker are not selected.
        other = dict(self.env, **{iso.AGENT_MARKER_ENV: "agent-someone-else"})
        own_child = subprocess.Popen(["sleep", "300"], cwd=self.actor, env={"PATH": os.defpath})
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], env=other,
                                   start_new_session=True)
        for process in (own_child, foreign):
            self.addCleanup(process.wait)
            self.addCleanup(process.kill)
        time.sleep(0.3)
        scan = iso.processes_with_marker(self.instance, cwd_roots=[self.actor])
        self.assertFalse({own_child.pid, foreign.pid} & set(scan.pids), scan.to_dict())

    def test_scan_failure_and_survivors_make_the_receipt_unconfirmed(self):
        failed = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance,
                                               scan=mock.Mock(side_effect=OSError("ps")))
        self.assertFalse(failed.confirmed)
        self.assertEqual(failed.errors[0], "marker_scan:OSError")
        survivor = iso.MarkedProcess(4242, 1, 4242, "env")
        stubborn = iso.MarkerScan(self.instance, (survivor,), (survivor.info(),))
        with mock.patch.object(iso, "_alive", return_value=True), mock.patch.object(iso.os, "killpg"):
            report = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance, grace_s=0, verify_s=0,
                                                   scan=lambda *args, **kwargs: stubborn, max_rounds=2)
        self.assertFalse(report.confirmed)
        self.assertEqual(report.remaining, (4242,))
        self.assertEqual(len(report.rounds), 2)
        self.assertEqual(report.to_dict()["confirmed"], False)

    def test_group_signalling_rules(self):
        # cwd-only selection in a group that also holds an unselected process: signalled by pid, never killpg.
        lonely = iso.MarkedProcess(5001, 1, 6000, "cwd")
        grouped = iso.MarkedProcess(5002, 1, 7000, "cwd")
        marked = iso.MarkedProcess(5003, 1, 8000, "env")
        table = (lonely.info(), iso.ProcessInfo(5009, 1, 6000), grouped.info(), marked.info(),
                 iso.ProcessInfo(5010, 5003, 8000))
        scans = [iso.MarkerScan(self.instance, (lonely, grouped, marked), table), iso.MarkerScan(self.instance, (), ())]
        with mock.patch.object(iso, "_alive", return_value=False), mock.patch.object(iso.os, "killpg") as killpg, \
                mock.patch.object(iso.os, "kill") as kill:
            report = iso.terminate_agent_processes(iso.TreeSnapshot(0, ()), self.instance, grace_s=0, verify_s=0,
                                                   scan=lambda *args, **kwargs: scans.pop(0))
        self.assertTrue(report.confirmed)
        self.assertEqual(sorted(report.termination.signalled_groups), [7000, 8000])
        self.assertEqual(report.termination.signalled_pids, (5001,))
        self.assertEqual(sorted(c.args[0] for c in killpg.call_args_list), [7000, 8000])
        self.assertEqual([c.args[0] for c in kill.call_args_list], [5001])

    def test_terminate_tree_signal_groups_parameter(self):
        snapshot = iso.TreeSnapshot(77, (iso.ProcessInfo(77, 1, 77), iso.ProcessInfo(78, 77, 90)))
        with mock.patch.object(iso, "_alive", return_value=False), mock.patch.object(iso.os, "killpg") as killpg, \
                mock.patch.object(iso.os, "kill") as kill:
            report = iso.terminate_tree(snapshot, grace_s=0, verify_s=0, signal_groups={77})
        self.assertEqual((report.signalled_groups, report.signalled_pids), ((77,), (78,)))
        self.assertEqual([c.args for c in killpg.call_args_list], [(77, signal.SIGTERM)])
        self.assertEqual([c.args for c in kill.call_args_list], [(78, signal.SIGTERM)])

if __name__ == "__main__":
    unittest.main()
