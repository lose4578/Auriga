"""Install a built wheel outside the checkout and verify runtime entry points."""
from pathlib import Path
import subprocess
import sys
import tempfile


CHECK = r'''
import hashlib
import json
from pathlib import Path
import runpy
import sys

installation = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installation))
import auriga
from auriga.apps.library.static_assets import static_directory
from auriga.core.config import Config
from auriga.apps.evaluation.report import build_report
from auriga.apps.reference.reference_api import REFERENCE_ROOT
from auriga.cortex.context.public_task_rules import CATALOG_PATH, public_task_rules
from auriga.core.storage.store import Store
from auriga.cortex.session.actor_source import snapshot_actor_source
from auriga.cortex.session.codex_sandbox import CodexSandbox

package = Path(auriga.__file__).resolve().parent
assert package.is_relative_to(installation), package
snapshot = snapshot_actor_source(Path("published-harness"))
assert (snapshot / "auriga/runtime/tools/definitions.py").is_file()
assert not (snapshot / "auriga/robot/backends/robodojo/robot_worker.py").exists()
assert (static_directory() / "index.html").is_file()
assert (static_directory() / "app.js").is_file()
assert (package / "runtime/policy/policy_workers/pi05_arx_manifest.json").is_file()
assert (package / "runtime/policy/policy_workers/pi05_arx_client.py").is_file()
assert (package / "core/default_config.json").is_file()
catalog = json.loads(CATALOG_PATH.read_text())
assert catalog['version'] == 2 and len(catalog['tasks']) == 54
for task, entry in catalog['tasks'].items():
    rules = public_task_rules(task)
    assert rules['text'] == '\n\n'.join(entry['public_rules'])
    assert '| Score | Condition |' in rules['text']
config = Config.load(environ={})
assert config.queue.max_parallel == 60
assert (config.model.name, config.model.effort) == ("gpt-6.1-sol", "high")
assert config.model.codex_home == str(Path("~/.codex").expanduser())
assert config.model.price_card is None
assert config.model.native_memories is True
assert config.features.sandbox_enabled is False
assert config.features.external_perception is True
assert config.features.held_perception is True
assert config.features.grip_non_empty_min_aperture_m == 0.0015
assert config.features.grip_rigid_min_aperture_m == 0.002
assert config.features.grip_wide_min_aperture_m == 0.003
assert config.agent.backend == "codex"
assert "agent" not in config.to_dict()
from auriga.cortex.agents.factory import create_session
from auriga.cortex.agents.claude_code.session import ClaudeCodeSession
from auriga.cortex.session.wire.anthropic_messages import AnthropicMessagesWire
assert AnthropicMessagesWire().name == "anthropic_messages"
assert Config.from_dict({"agent": {"backend": "claude_code"}}).agent.backend == "claude_code"
archived = Config.from_dict({"version": 1})
assert (archived.model.name, archived.model.effort) == ("gpt-6-astra", "high")
assert archived.model.codex_home == str(Path("~/.codex").expanduser())
assert archived.model.price_card is None
assert archived.model.native_memories is True
operator_config = Path("operator.json")
operator_config.write_text(json.dumps({"model": {"name": "operator-model"}}))
operator = Config.load(environ={"AURIGA_CONFIG": str(operator_config)})
assert (operator.model.name, operator.model.effort) == ("operator-model", "high")
assert operator.model.codex_home == config.model.codex_home
assert operator.model.price_card is None
feature_config = Path("feature-only.json")
feature_config.write_text(json.dumps({"features": {"enable_wait": False, "sandbox_enabled": False}}))
feature_only = Config.load(feature_config, environ={"AURIGA_CONFIG": "missing.json"})
assert feature_only.model == config.model
assert feature_only.features.enable_wait is False
assert feature_only.features.sandbox_enabled is False
legacy_config = Path("legacy.json")
legacy_config.write_text(json.dumps({"model": {"name": "gpt-6-astra", "effort": "xhigh", "codex_home": "archived-auth", "price_card": None}}))
legacy = Config.load(legacy_config, environ={"AURIGA_CONFIG": str(operator_config)})
assert (legacy.model.name, legacy.model.effort) == ("gpt-6-astra", "xhigh")
assert legacy.model.codex_home == str((legacy_config.parent / "archived-auth").resolve())
assert legacy.model.price_card is None
provenance = json.loads((REFERENCE_ROOT / "third_party/roboprobe/api-provenance.json").read_text())
for name, record in provenance["files"].items():
    assert hashlib.sha256((REFERENCE_ROOT / provenance["vendored_root"] / name).read_bytes()).hexdigest() == record["sha256"]
store = Store("smoke-records", {})
reference = store.put_artifact(b"package integrity", "test", "text/plain")
assert store.resolve_artifact(reference)[1] == b"package integrity"
store.close()
for module in ("auriga.apps.gateway", "auriga.apps.evaluation.queue", "auriga.apps.evaluation.report", "auriga.apps.evaluation.audit"):
    sys.argv = [module, "--help"]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as result:
        assert result.code in (None, 0), (module, result.code)
print("Installed wheel: config defaults and precedence, archived config, imports, CLI, Web, policy, pinned reference and Store checks passed")
'''


if __name__ == "__main__":
    wheel = Path(sys.argv[1]).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="auriga-installed-") as directory:
        installation = Path(directory) / "site-packages"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installation), str(wheel)], check=True)
        subprocess.run([sys.executable, "-I", "-c", CHECK, str(installation)], cwd=directory, check=True)
