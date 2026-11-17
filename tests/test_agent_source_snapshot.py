"""Workbench publishes the Harness source snapshot for every backend before the session starts (design R4.4)."""
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auriga.apps.gateway.workbench import Workbench
from auriga.core.config import Config
from auriga.cortex.agents.base import AgentCapabilities
from auriga.cortex.session.actor_source import SOURCE_FILES
from test_agent_workbench import fake_session, fixture_harness, terminal_double


class WorkbenchSourceSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_codex_backend_gets_the_snapshot_named_by_its_instructions(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        create, created = fake_session(AgentCapabilities())
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        with patch("auriga.apps.gateway.workbench.create_session", side_effect=create):
            workbench = Workbench(Path(temporary.name) / "run", robot=robot, visual_review_model=None,
                                  config=Config.from_dict({"features": {"self_briefing": False, "terminal": False,
                                                                        "sandbox_enabled": False}}))
        session, = created
        workbench.harness = fixture_harness(workbench)
        snapshot = workbench.root / "private" / "actor-source"
        seen = {}
        original_start = session.start

        def start(tools, instructions, resume_thread_id=None):
            # The copy exists before the backend starts, so a sandbox can mount it at /harness.
            seen["files"] = sorted(path.relative_to(snapshot).as_posix()
                                   for path in snapshot.rglob("*") if path.is_file())
            seen["instructions"] = instructions
            return original_start(tools, instructions, resume_thread_id)
        session.start = start
        with patch("auriga.apps.gateway.workbench.Terminal", return_value=terminal_double()):
            await workbench.start("Smoke instruction.")
        self.addAsyncCleanup(workbench.close)
        self.assertEqual(seen["files"], sorted("auriga/" + name for name in SOURCE_FILES))
        self.assertIn(f"Read the public Harness implementation under {snapshot}.", seen["instructions"])
        self.assertEqual(stat.S_IMODE((workbench.root / "private").stat().st_mode), 0o700)
        self.assertFalse((snapshot / "auriga/robot/backends/robodojo").exists())


if __name__ == "__main__":
    unittest.main()
