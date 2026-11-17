"""Gateway callers of the AgentSession protocol, exercised with a non-Codex FakeAgentSession and a Codex-shaped run."""
import asyncio
import fcntl
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from auriga.apps.gateway.server import serve
from auriga.apps.gateway.workbench import Workbench
from auriga.core.config import Config
from auriga.core.observation_policy import ACTOR_PROFILE
from auriga.core.storage.store import Store
from auriga.cortex.agents.base import AgentCapabilities
from auriga.cortex.session.codex import CODEX_CLI_VERSION, CODEX_SDK_VERSION, CodexSession
from support.fake_agent import FakeAgentSession
from test_harness import FixtureRobot

# Top-level manifest keys a Codex run wrote at 9643521, before the agent protocol (fixture start, no self-briefing).
CODEX_MANIFEST_KEYS_BEFORE = {
    'schema_version', 'run_id', 'episode_id', 'env_instance_id', 'env_generation', 'data_origin',
    'actor_information_policy', 'motion_review', 'vision_compare', 'task_planning', 'grasp_memory_dir',
    'observation_profile', 'clock_policy', 'model', 'provider', 'effort', 'price_card', 'codex_sdk', 'codex_cli',
    'codex_cli_executable', 'capabilities', 'image_settings', 'model_image_policy', 'actor_payload_profile',
    'enable_wait', 'tracking_guard', 'joint_replan', 'task', 'seed', 'layout_id', 'max_actions', 'actor_wait_budget_s',
    'runtime_protocol', 'model_request_limit', 'model_request_count_basis', 'model_attempt_max_retries',
    'model_request_timeout_s', 'model_auto_retries', 'model_retry_policy', 'no_progress_timeout_s', 'startup_timeout_s',
    'progress_clock', 'read_only_queries_between_actions', 'pi05_enabled', 'visual_review_model', 'source_sha256',
    'benchmark', 'benchmark_terminal_policy', 'actor_stall_policy', 'actor_progress_policy', 'scoring_policy',
    'completion_guidance', 'source_ref', 'task_notes', 'task_notes_source', 'task_notes_config', 'handoff_policy',
    'public_task_rules', 'task_instruction_override', 'thread_id', 'app_server_pid', 'terminal_session_id',
    'terminal_pid', 'effective_config', 'tool_catalog', 'local_scene'}
CODEX_ONLY_MANIFEST_KEYS = ('codex_sdk', 'codex_cli', 'codex_cli_executable', 'app_server_pid')


def fixture_harness(workbench):
    """Process-free harness double with the real registry, as the existing startup tests build it."""
    done = threading.Event()
    done.set()
    harness = SimpleNamespace(
        accepting=True, active_action=None, states={}, action_count=0,
        latest={"task_instruction": "Fixture native task.", "observation_id": "initial",
                "observation_profile": ACTOR_PROFILE},
        capabilities={}, episode_done=threading.Event(), episode_terminal=None,
        native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(),
        scene=SimpleNamespace(capabilities=lambda: {}), start=Mock(),
        action_budget=lambda: dict(native_steps_used=0, native_step_limit=700, native_steps_remaining=700))
    for field in ('tool_registry', 'config', 'disabled_tools', 'observation_profile', 'motion_review', 'enable_wait',
                  'vision_compare', 'visual_reviewer', 'action_policy', 'programs'):
        setattr(harness, field, getattr(workbench.harness, field))
    harness.request_stop = lambda: setattr(harness, "accepting", False)
    return harness


def fake_session(capabilities=AgentCapabilities(), **options):
    created = []

    def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
        session = FakeAgentSession(run_dir, store, tool_handler, capabilities=capabilities, price_card=price_card,
                                   **options)
        created.append(session)
        return session
    return create, created


def terminal_double():
    return SimpleNamespace(start=AsyncMock(), close=AsyncMock(), session_id="fixture-terminal",
                           process=SimpleNamespace(pid=-1), status=lambda: dict(mode="read_only"))


class FakeBackendStartupTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, capabilities, attach_spec=None, terminal=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        create, created = fake_session(capabilities)
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        with patch("auriga.apps.gateway.workbench.create_session", side_effect=create):
            workbench = Workbench(Path(temporary.name) / "run", robot=robot, visual_review_model=None,
                                  config=Config.from_dict({'features': {'self_briefing': False, 'terminal': terminal}}))
        session, = created
        session.attach_spec = attach_spec
        self.assertIs(workbench.session, session)
        self.assertEqual(workbench.harness.delivery_channel, "agent_tool")
        self.assertIs(session.on_disconnect.__self__, workbench.harness)
        workbench.harness = fixture_harness(workbench)
        terminal = terminal_double()
        with patch("auriga.apps.gateway.workbench.Terminal", return_value=terminal) as create_terminal:
            await workbench.start("Smoke instruction.")
        self.addAsyncCleanup(workbench.close)
        return workbench, session, create_terminal, terminal

    async def test_backend_without_terminal_skips_it_and_records_agent_backend(self):
        workbench, session, create_terminal, _ = await self.start(AgentCapabilities())
        create_terminal.assert_not_called()
        self.assertIsNone(workbench.terminal)
        self.assertEqual(session.calls[:2], [("start", len(workbench.actor_tools()), None), ("run_turn", "Smoke instruction.")])
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertEqual(manifest["agent_backend"], session.provenance())
        for key in CODEX_ONLY_MANIFEST_KEYS:
            self.assertNotIn(key, manifest)
        self.assertEqual((manifest["terminal_session_id"], manifest["terminal_pid"]), (None, None))
        self.assertEqual((manifest["model"], manifest["thread_id"]), (session.model, session.thread_id))
        snapshot = workbench.snapshot()
        self.assertEqual((snapshot["agent_backend"], snapshot["agent_status"], snapshot["codex_status"]),
                         ("fake_agent", "running", "running"))

    async def test_terminal_capable_backend_attaches_with_its_spec(self):
        spec = dict(cwd=Path("/fixture/actor"), command_prefix=["fixture-tui", "--read-only"], env={"A": "1"},
                    attach_uri="ws://127.0.0.1:1/fixture", thread_id="fixture-thread")
        workbench, _, create_terminal, terminal = await self.start(AgentCapabilities(terminal_attach=True), spec)
        create_terminal.assert_called_once_with(workbench.root, spec["cwd"], command_prefix=spec["command_prefix"],
                                                env=spec["env"])
        terminal.start.assert_awaited_once_with(spec["attach_uri"], spec["thread_id"])
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertEqual(manifest["terminal_session_id"], "fixture-terminal")

    async def test_terminal_feature_off_never_asks_the_session(self):
        workbench, _, create_terminal, _ = await self.start(AgentCapabilities(terminal_attach=True),
                                                            dict(cwd="x"), terminal=False)
        create_terminal.assert_not_called()


class ControlFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        robot = FixtureRobot(root)
        robot.close = lambda: None
        self.create, self.created = fake_session(self.capabilities())
        config = Config.from_dict({'features': {'grasp_memory_dir': str(root / 'memory'),
                                                'observation_profile': 'rgb_proprio_v1', 'motion_review': False}})
        with patch("auriga.apps.gateway.workbench.create_session", side_effect=self.create):
            self.workbench = Workbench(root / 'run', robot=robot, config=config)
        self.session, = self.created
        self.workbench.harness.start()
        self.workbench.starting, self.workbench.status = False, 'running'
        self.session.start([], 'instructions')
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        self.origin = str(self.client.make_url('')).rstrip('/')
        owner = await (await self.client.post('/v1/control', headers={'Origin': self.origin}, json=dict(
            request_id='claim', command='claim_input', client_id='fixture'))).json()
        self.lease = dict(client_id='fixture', writer_token=owner['writer_token'])

    async def asyncTearDown(self):
        await self.client.close()
        await self.workbench.close()

    @staticmethod
    def capabilities():
        return AgentCapabilities()

    async def control(self, identifier, command, **fields):
        response = await self.client.post('/v1/control', headers={'Origin': self.origin}, json=dict(
            request_id=identifier, command=command, **self.lease, **fields))
        return response.status, await response.json()


class FakeBackendControlTests(ControlFixture):
    async def test_definite_rejection_and_unknown_admission_map_like_codex_errors(self):
        self.session.script_admission('reject', code='http_400', message='fixture rejection')
        status, body = await self.control('rejected', 'prompt', prompt='task')
        self.assertEqual((status, body['status'], body['reason'], body['native_rpc_code']),
                         (400, 'rejected', 'fixture rejection', 'http_400'))
        self.assertEqual(self.session.state, 'idle')
        self.assertTrue(self.workbench.harness.accepting)
        self.session.script_admission('reject', code=429, message='overloaded', busy=True)
        status, body = await self.control('busy', 'prompt', prompt='task')
        self.assertEqual((status, body['status']), (429, 'rejected'))
        self.assertEqual(await self.control('busy', 'prompt', prompt='task'), (status, body))
        self.session.script_admission('unknown')
        status, body = await self.control('lost', 'prompt', prompt='task')
        self.assertEqual((status, body['status']), (409, 'unknown'))
        self.assertEqual((self.session.state, self.workbench.status), ('disconnected', 'unknown'))
        self.assertFalse(self.workbench.harness.accepting)
        self.assertEqual(sum(call[0] == 'run_turn' for call in self.session.calls), 3)

    async def test_steer_is_refused_when_the_backend_cannot_join_the_active_turn(self):
        status, body = await self.control('first', 'prompt', prompt='task')
        self.assertEqual((status, body['status']), (200, 'running'))
        status, body = await self.control('steer', 'steer', prompt='operator text')
        self.assertEqual((status, body['error']), (400, 'steer_unsupported_by_backend'))
        self.assertNotIn(('steer', 'operator text'), self.session.calls)



class SteeringBackendControlTests(ControlFixture):
    @staticmethod
    def capabilities():
        return AgentCapabilities(steer=True)

    async def test_steer_joins_the_active_turn_when_capable(self):
        await self.control('first', 'prompt', prompt='task')
        status, body = await self.control('steer', 'steer', prompt='operator text')
        self.assertEqual((status, body['status'], body['turn_id']), (200, 'steered', self.session.turn_id))
        self.assertIn(('steer', 'operator text'), self.session.calls)


class AbortStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_abort_signals_every_process_the_session_owns(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        session = FakeAgentSession()
        session.processes = [('agent_cli', process), ('agent_helper', None)]
        events = []
        owner = SimpleNamespace(robot=SimpleNamespace(process=None), session=session, terminal=None,
                                store=SimpleNamespace(event=lambda kind, payload: events.append((kind, payload))))
        starting = asyncio.create_task(asyncio.to_thread(process.wait))
        try:
            await asyncio.wait_for(Workbench.abort_startup(owner, starting), timeout=8)
        finally:
            if process.poll() is None:
                process.kill()
            await starting
        signalled = [payload for kind, payload in events if kind == 'run.startup_process_signalled']
        self.assertEqual([(item['role'], item['pid'], item['signal']) for item in signalled],
                         [('agent_cli', process.pid, 'SIGTERM')])
        exits = [payload for kind, payload in events if kind == 'run.startup_process_exit']
        self.assertTrue(exits[0]['process_exit_confirmed'])


class ReplayAndResumeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / 'run'

    @staticmethod
    def codex_double(workbench):
        """Duck-typed, process-free Codex session as the existing startup fixtures build it."""
        return SimpleNamespace(
            thread_id="recorded-thread", turn_id="first-turn", state="completed", turn_done=threading.Event(),
            model="fixture", provider="fixture", effort="high", price_card=None, cli_executable=Path("/fixture/codex"),
            command_prefix=[], actor_dir=workbench.root, socket_path=workbench.root / "fixture.sock", env={},
            process=SimpleNamespace(pid=-1), start=Mock(return_value="recorded-thread"), run_turn=Mock(), close=Mock(),
            model_proxy=None)

    async def recorded_run(self, agent=None, codex=None):
        """A completed Codex-shaped run (fixture start) whose runtime/codex is ``codex`` (default: a recorded thread),
        optionally with a runtime/agent record."""
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        workbench = Workbench(self.root, robot=robot, visual_review_model=None)
        workbench.harness = fixture_harness(workbench)
        workbench.session = self.codex_double(workbench)
        with patch("auriga.apps.gateway.workbench.Terminal", return_value=terminal_double()):
            await workbench.start("unused", benchmark=True)
        workbench.store.set_state('runtime', 'codex', dict(thread_id='recorded-thread', app_server_instance_id='a-1')
                                  if codex is None else codex)
        workbench.store.set_state('runtime', 'capabilities', dict(data_origin='synthetic_fixture'))
        if agent is not None:
            workbench.store.set_state('runtime', 'agent', agent)
        await workbench.close()

    async def test_resume_refuses_a_changed_agent_backend(self):
        await self.recorded_run(dict(backend='claude_code', thread_id='session-1', instance_id='agent-1'))
        robot = SimpleNamespace(close=Mock())
        try:
            Workbench(self.root, robot=robot, resume=True)
        except ValueError as error:
            # Keep the traceback, and with it the refused Workbench, alive: only an explicit release may free
            # the run (garbage collection of the half-built object must not make this test pass).
            refusal = error
        else:
            self.fail('resume with a changed agent backend was accepted')
        self.assertRegex(str(refusal), '^resume_agent_backend_changed: .*claude_code')
        frame = refusal.__traceback__
        while frame is not None and not isinstance(frame.tb_frame.f_locals.get('self'), Workbench):
            frame = frame.tb_next
        refused = frame.tb_frame.f_locals['self']
        self.assertTrue(refused.run_lock.closed)
        with self.assertRaises(sqlite3.ProgrammingError):
            refused.store.db.execute('SELECT 1')
        # The gateway lock is free: another owner can take it without blocking.
        with (self.root / 'gateway.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        del refusal, frame, refused

    async def test_resume_refuses_a_codex_run_under_another_requested_backend(self):
        await self.recorded_run()
        config = Config.load().with_overrides({'agent': {'backend': 'opencode'}})
        with self.assertRaisesRegex(ValueError, '^resume_agent_backend_changed: .*codex'):
            Workbench(self.root, robot=SimpleNamespace(close=Mock()), resume=True, config=config)

    async def resume_start(self):
        """Resume the recorded run with a Codex-shaped session double; return (workbench, session)."""
        workbench = Workbench(self.root, robot=SimpleNamespace(close=Mock()), resume=True)
        self.addAsyncCleanup(workbench.close)
        workbench.harness = fixture_harness(workbench)
        session = workbench.session = self.codex_double(workbench)
        with patch("auriga.apps.gateway.workbench.Terminal", return_value=terminal_double()):
            await workbench.start("unused")
        return workbench, session

    async def test_resume_of_a_codex_archive_uses_the_recorded_thread(self):
        await self.recorded_run()
        workbench, session = await self.resume_start()
        self.assertEqual(workbench.previous_agent['backend'], 'codex')
        self.assertEqual(workbench.previous_agent['thread_id'], 'recorded-thread')
        self.assertEqual(workbench.previous_agent['instance_id'], 'a-1')
        session.start.assert_called_once()
        self.assertEqual(session.start.call_args.args[2:], ('recorded-thread',))
        session.run_turn.assert_not_called()

    async def test_resume_with_an_empty_native_record_starts_a_fresh_thread(self):
        # As before the protocol: an empty runtime/codex is no recorded thread.
        await self.recorded_run(codex={})
        workbench, session = await self.resume_start()
        self.assertIsNone(workbench.previous_agent)
        session.start.assert_called_once()
        self.assertEqual(len(session.start.call_args.args), 2)

    async def test_replay_enters_replay_through_the_session(self):
        await self.recorded_run(dict(backend='fake_agent', thread_id='fake-recorded-thread', instance_id='agent-1'))
        create, created = fake_session()
        with patch("auriga.apps.gateway.workbench.create_session", side_effect=create):
            replay = Workbench(self.root, robot=SimpleNamespace(close=Mock()), replay=True)
        session, = created
        session.start = Mock(side_effect=AssertionError('replay must not start an agent'))
        try:
            await replay.start('must not initiate a turn')
            self.assertEqual((session.state, session.thread_id), ('replay', 'fake-recorded-thread'))
            self.assertEqual(replay.snapshot()['agent_status'], 'replay')
            self.assertEqual(session.calls, [])
        finally:
            await replay.close()


class CodexManifestTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_manifest_only_gains_the_agent_backend_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
            workbench = Workbench(Path(temporary) / 'run', robot=robot, visual_review_model=None,
                                  config=Config.from_dict({'features': {'self_briefing': False}}))
            self.assertIsInstance(workbench.session, CodexSession)
            self.assertEqual(workbench.harness.delivery_channel, 'codex_tool')
            workbench.harness = fixture_harness(workbench)
            # Duck-typed Codex session without protocol members (as ~15 existing fixtures build it).
            session = workbench.session = SimpleNamespace(
                thread_id="fixture-thread", turn_id=None, state="completed", turn_done=threading.Event(),
                model="fixture", provider="fixture", effort="high", price_card=None, cli_executable=Path("/fixture/codex"),
                command_prefix=["fixture-sandbox"], actor_dir=workbench.root, socket_path=workbench.root / "fixture.sock",
                env={"HOME": "/actor"}, process=SimpleNamespace(pid=4242), start=Mock(return_value="fixture-thread"),
                run_turn=Mock(), close=Mock(), model_proxy=None)
            terminal = terminal_double()
            with patch("auriga.apps.gateway.workbench.Terminal", return_value=terminal) as create_terminal:
                await workbench.start("Smoke instruction.")
            try:
                create_terminal.assert_called_once_with(workbench.root, workbench.root, env={"HOME": "/actor"},
                    command_prefix=["fixture-sandbox", "/fixture/codex", "-c", 'model="fixture"',
                                    "-c", 'model_reasoning_effort="high"'])
                terminal.start.assert_awaited_once_with(f"unix://{workbench.root / 'fixture.sock'}", "fixture-thread")
                manifest = json.loads((workbench.root / 'manifest.json').read_text())
                self.assertEqual(set(manifest) - CODEX_MANIFEST_KEYS_BEFORE, {'agent_backend'})
                self.assertEqual(CODEX_MANIFEST_KEYS_BEFORE - set(manifest), set())
                self.assertEqual((manifest['codex_sdk'], manifest['codex_cli'], manifest['codex_cli_executable'],
                                  manifest['app_server_pid']), (CODEX_SDK_VERSION, CODEX_CLI_VERSION, '/fixture/codex', 4242))
                self.assertEqual(manifest['agent_backend'], CodexSession.provenance(session))
                self.assertEqual(manifest['agent_backend']['kind'], 'codex')
                self.assertNotIn('agent', manifest['effective_config'])
                effective = json.loads((workbench.root / 'effective_config.json').read_text())
                self.assertNotIn('agent', effective)
                snapshot = workbench.snapshot()
                self.assertEqual((snapshot['agent_backend'], snapshot['agent_status'], snapshot['codex_status']),
                                 ('codex', 'completed', 'completed'))
            finally:
                await workbench.close()


class ServerSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def summary(self, session):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = dict(run_id="fixture", episode_id="episode", env_instance_id="env", env_generation=1)
            store = Store(root, identity)

            def close():
                store.close()
                return {"steps": {}, "errors": [], "robot": {"native_close_acknowledged": True}}
            workbench = SimpleNamespace(
                store=store, identity=identity, session=session, status="running",
                harness=SimpleNamespace(action_count=0, active_action=None, episode_done=threading.Event(),
                                        episode_terminal=None, native_action_count=0, started_at=time.monotonic()),
                app=web.Application, snapshot=lambda: {"status": "running"},
                start=AsyncMock(), stop=AsyncMock(), close=AsyncMock(side_effect=close),
                current_unknown_actions=lambda: False, control_lock=asyncio.Lock(), has_active_programs=lambda: False,
                observer_evaluate=Mock(return_value={}), observer_video=Mock(return_value={}))
            args = SimpleNamespace(run_dir=str(root), resume=False, replay=False, price_card=None, category="diagnostic",
                                   task="fixture", title="fixture", tag=[], gpu=0, max_actions=1,
                                   observation_profile="rgb_proprio_v1", seed=0, layout_id=0, port=0, prompt="fixture",
                                   task_notes=None, task_notes_source=None, wall_budget_s=1, exit_after_turn=True,
                                   benchmark=False)
            with patch("auriga.apps.gateway.server.Workbench", return_value=workbench), \
                    patch("auriga.apps.gateway.server.print"):
                await serve(args)
            return json.loads((root / "summary.json").read_text())

    def failed_session(self, **options):
        session = FakeAgentSession(**options)
        session.thread_id, session.state = "thread", "failed"
        session.turn_result = {"id": "turn", "status": "failed", "error": {"message": "fixture failure"}}
        session.turn_done.set()
        return session

    async def test_non_codex_failure_uses_neutral_names_and_summary_fields(self):
        summary = await self.summary(self.failed_session())
        self.assertEqual(summary["infrastructure_error"], "agent_turn_failed: fixture failure")
        self.assertEqual((summary["agent_backend"], summary["agent_turn_status"], summary["codex_turn_status"]),
                         ("fake_agent", "failed", "failed"))

    async def test_codex_kind_keeps_historical_names(self):
        summary = await self.summary(self.failed_session(backend_kind="codex"))
        self.assertEqual(summary["infrastructure_error"], "codex_turn_failed: fixture failure")
        self.assertEqual(summary["agent_backend"], "codex")


if __name__ == '__main__':
    unittest.main()
