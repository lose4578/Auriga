"""Self-tests of ``tests/support/fake_claude_sdk.py`` (the scripted ``ClaudeSDKClient`` stand-in)."""
import asyncio
import inspect
import threading
import unittest
from pathlib import Path

from support.fake_claude_sdk import (SDK_AVAILABLE, FakeClaudeSDK, FakeClaudeSDKClient, assistant_message,
                                     echo_message, init_message, load_capture, message_from_capture, requires_sdk,
                                     result_message, state_message, tool_result_message, tool_use, turn_messages)

FIXTURES = Path(__file__).parent / "fixtures" / "claude_sdk"


class Loop:
    """A private event loop on a thread, like the session's ``claude-loop``."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro, timeout=5):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


async def drain(client, into):
    async for message in client.receive_messages():
        into.append(message)


@requires_sdk
class FakeClientCheck(unittest.TestCase):
    def setUp(self):
        self.loop = Loop()
        self.fake = FakeClaudeSDK()
        self.client = self.fake.factory(None)

    def tearDown(self):
        self.loop.close()

    def connect(self):
        self.loop.run(self.client.connect())

    def test_unconnected_calls_raise_like_the_sdk(self):
        from claude_agent_sdk._errors import CLIConnectionError
        for call in (self.client.query("x"), self.client.interrupt(), self.client.set_model("m")):
            with self.assertRaisesRegex(CLIConnectionError, "Not connected"):
                self.loop.run(call)
        with self.assertRaisesRegex(CLIConnectionError, "Not connected"):
            self.loop.run(drain(self.client, []))

    def test_stream_in_order_then_clean_end(self):
        self.connect()
        received = []
        reader = self.loop.submit(drain(self.client, received))
        self.client.emit(*turn_messages("hello", session_id="s1"))
        self.client.end()
        reader.result(5)
        self.assertEqual([type(m).__name__ for m in received],
                         ["SystemMessage", "SystemMessage", "AssistantMessage", "ResultMessage", "SystemMessage"])
        self.assertEqual(received[-1].data["state"], "idle")

    def test_failure_raised_once_and_ends_stream(self):
        from claude_agent_sdk._errors import ProcessError
        self.connect()
        received = []
        reader = self.loop.submit(drain(self.client, received))
        self.client.emit(assistant_message("a"))
        self.client.kill_process(-9)
        with self.assertRaises(ProcessError):
            reader.result(5)
        self.assertEqual(len(received), 1)
        self.assertEqual((self.client.process.returncode,), (-9,))

    def test_second_concurrent_consumer_is_a_test_error(self):
        self.connect()
        first = self.loop.submit(drain(self.client, []))
        self.client.wait_for_call("connect")
        for _ in range(200):
            if self.client._consumer_active:
                break
            threading.Event().wait(0.01)
        with self.assertRaisesRegex(RuntimeError, "fake_second_receive_consumer"):
            self.loop.run(drain(self.client, []))
        self.client.end()
        first.result(5)

    def test_disconnect_is_idempotent_ends_stream_and_blocks_later_calls(self):
        from claude_agent_sdk._errors import CLIConnectionError
        self.connect()
        reader = self.loop.submit(drain(self.client, []))
        self.loop.run(self.client.disconnect())
        self.loop.run(self.client.disconnect())
        reader.result(5)
        with self.assertRaises(CLIConnectionError):
            self.loop.run(self.client.query("late"))
        self.assertEqual(self.client.call_names(), ["connect", "disconnect", "disconnect"])

    def test_query_records_and_runs_handlers_that_emit(self):
        self.connect()
        received = []
        reader = self.loop.submit(drain(self.client, received))
        self.client.on_query.append(lambda client, text: client.emit(echo_message(text)))
        self.loop.run(self.client.query("go", "sid"))
        self.assertEqual(self.client.queries(), ["go"])
        self.assertEqual(self.client.calls[1], ("query", "go", "sid"))
        self.client.end()
        reader.result(5)
        self.assertEqual(received[0].content, "go")

    def test_fault_injection_and_gates(self):
        self.connect()
        self.client.query_error = lambda index: OSError("write failed") if index == 2 else None
        self.loop.run(self.client.query("one"))
        with self.assertRaisesRegex(OSError, "write failed"):
            self.loop.run(self.client.query("two"))
        self.loop.run(self.client.query("three"))
        self.client.hold("interrupt")
        pending = self.loop.submit(self.client.interrupt())
        self.assertTrue(self.client.wait_for_call("interrupt"))
        threading.Event().wait(0.05)
        self.assertFalse(pending.done())
        self.client.release("interrupt")
        pending.result(5)
        self.client.set_model_error = RuntimeError("Control request timeout: set_model")
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.loop.run(self.client.set_model("m2"))

    def test_failing_connect_disconnects_before_raising(self):
        self.client.connect_error = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.loop.run(self.client.connect())
        self.assertEqual(self.client.call_names(), ["connect", "disconnect"])
        with self.assertRaisesRegex(RuntimeError, "fake_client_not_connected"):
            self.client.emit(assistant_message("x"))

    def test_emit_from_other_thread_and_init_on_connect(self):
        self.fake.mcp_tools = ("robot_observe",)
        self.fake.emit_init_on_connect()
        from claude_agent_sdk import ClaudeAgentOptions
        client = self.fake.factory(ClaudeAgentOptions(tools=["Bash"], session_id="sid-1", model="mm"))
        self.loop.run(client.connect())
        received = []
        reader = self.loop.submit(drain(client, received))
        thread = threading.Thread(target=lambda: (client.emit(assistant_message("t")), client.end()))
        thread.start()
        thread.join()
        reader.result(5)
        init = received[0]
        self.assertEqual((init.subtype, init.data["session_id"], init.data["model"], init.data["tools"]),
                         ("init", "sid-1", "mm", ["Bash", "mcp__auriga__robot_observe"]))
        self.assertIs(self.fake.client, client)

    def test_process_double_has_signal_semantics(self):
        process = self.client.process
        self.assertIsNone(process.returncode)
        process.terminate()
        process.kill()
        self.assertEqual((process.signals, process.returncode), (["TERM", "KILL"], -15))

    def test_public_surface_matches_the_installed_sdk_client(self):
        from claude_agent_sdk import ClaudeSDKClient
        for name in ("connect", "receive_messages", "query", "interrupt", "set_model", "disconnect"):
            real, fake = inspect.signature(getattr(ClaudeSDKClient, name)), inspect.signature(
                getattr(FakeClaudeSDKClient, name))
            self.assertEqual(list(real.parameters), list(fake.parameters), name)
            self.assertEqual(inspect.iscoroutinefunction(getattr(ClaudeSDKClient, name)) or
                             inspect.isasyncgenfunction(getattr(ClaudeSDKClient, name)),
                             inspect.iscoroutinefunction(getattr(FakeClaudeSDKClient, name)) or
                             inspect.isasyncgenfunction(getattr(FakeClaudeSDKClient, name)), name)


@requires_sdk
class BuildersAndCaptures(unittest.TestCase):
    def test_builders_make_real_sdk_dataclasses(self):
        import claude_agent_sdk as sdk
        messages = [init_message("s", tools=["Bash"]), state_message("running"),
                    assistant_message("hi", tool_use("t1", "Bash", command="ls")), tool_result_message("t1", "out"),
                    result_message(), echo_message("x")]
        self.assertEqual([type(m) for m in messages],
                         [sdk.SystemMessage, sdk.SystemMessage, sdk.AssistantMessage, sdk.UserMessage,
                          sdk.ResultMessage, sdk.UserMessage])
        self.assertEqual(messages[2].content[1].input, {"command": "ls"})
        self.assertEqual(messages[3].content[0].tool_use_id, "t1")

    def test_every_fixture_loads_into_sdk_objects(self):
        import claude_agent_sdk as sdk
        files = sorted(FIXTURES.glob("*.jsonl"))
        self.assertGreaterEqual(len(files), 7)
        for path in files:
            messages = load_capture(path)
            self.assertTrue(messages, path.name)
            self.assertTrue(all(isinstance(m, (sdk.SystemMessage, sdk.AssistantMessage, sdk.UserMessage,
                                               sdk.ResultMessage)) for m in messages), path.name)
        steer = load_capture(FIXTURES / "s3-during-tool-state-replay.jsonl")
        self.assertEqual([type(m).__name__ for m in steer][:3], ["SystemMessage", "SystemMessage", "UserMessage"])
        self.assertEqual(steer[-2].terminal_reason, "completed")
        self.assertEqual(steer[-1].data["state"], "idle")

    def test_message_from_capture_converts_blocks(self):
        message = message_from_capture(dict(_type="AssistantMessage", model="m", message_id="mid", usage={},
                                            content=[dict(_type="ToolUseBlock", id="i", name="Bash", input={})]))
        self.assertEqual(type(message.content[0]).__name__, "ToolUseBlock")


class Availability(unittest.TestCase):
    def test_sdk_is_installed_in_the_project_environment(self):
        self.assertTrue(SDK_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
