"""Command-line entry point for one workbench run."""

import argparse
import asyncio
import json

from ...core.presentation.actor_payload import ACTOR_PAYLOAD_PROFILES
from ...runtime.tools.builtins import create_registry, CONFIG_TOOL_NAMES
from ...core.observation_policy import ACTOR_PROFILES
from ...cortex.supervision.completion_guidance import COMPLETION_GUIDANCE_MODES
from ...core.config import Config, ConfigError, cli_overrides, namespace_values
from .server import serve
from .workbench import _configured_tool_specs


def main():
    parser = argparse.ArgumentParser(description="Run Astra + RoboDojo with a persistent native agent workbench", argument_default=argparse.SUPPRESS)
    parser.add_argument('--config', help='Unified JSON config; defaults to AURIGA_CONFIG or configs/auriga.json')
    parser.add_argument('--print-config', action='store_true', help='Print resolved configuration and exit without starting services')
    parser.add_argument('--model', help='Actor model name')
    parser.add_argument('--image-feedback-policy', choices=('all', 'adaptive'),
                        help='Automatic action camera delivery; explicit observations retain all requested views')
    parser.add_argument('--effort', help='Actor reasoning effort')
    parser.add_argument('--agent-backend', choices=('codex', 'claude_code', 'opencode'),
                        help='Override agent.backend: the agent runtime that drives the actor (default codex); other agent.* settings are config-file only')
    parser.add_argument('--codex-home', help='Source Codex configuration/auth directory (credentials are never embedded)')
    parser.add_argument('--grasp-memory-dir', help='Persistent reviewed grasp memory directory')
    parser.add_argument('--disabled-tools', nargs='*', help='Tool names blocked in both the model interface and program dispatch; empty clears config list')
    for name in ('record-video', 'export-video', 'terminal', 'depth-skills', 'depth-auto-feedback', 'depth-event-payload'):
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction)
    parser.add_argument("--run-dir")
    parser.add_argument("--experiment-root", help="Archive root; default ~/exp/auriga")
    parser.add_argument("--price-card",
                        help="JSON price schedule for an explicitly labelled estimate; defaults to AURIGA_PRICE_CARD")
    parser.add_argument("--category")
    parser.add_argument("--title")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--clear-tags", action="store_const", const=[], dest="tag", help="Clear tags inherited from the config file")
    parser.add_argument("--replay", action=argparse.BooleanOptionalAction, help="Offline read-only viewer: no GPU, app-server, model or actions")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("--observation-profile", choices=ACTOR_PROFILES,
                        help="Calibrated RGB/proprio by default; explicit rgb_proprio_v1 retains the legacy control boundary")
    parser.add_argument("--vision-compare", action=argparse.BooleanOptionalAction, help="Enable RGB feature comparison and comparison-based skills")
    parser.add_argument("--robotic-view", action=argparse.BooleanOptionalAction,
                        help="Expose read-only wrist-camera look-at pose suggestions (default: disabled)")
    parser.add_argument("--stereo-guide", action=argparse.BooleanOptionalAction,
                        help="Expose read-only epipolar guidance between calibrated RGB cameras (default: disabled)")
    parser.add_argument("--motion-review", action=argparse.BooleanOptionalAction,
                        help="Expose motion preview, reviewed execution and grasp planning tools; legacy observation profile disables these")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--layout-id", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int, help="Per-run loopback port; 0 chooses a free port. The permanent library uses 8780.")
    parser.add_argument("--max-actions", type=int, help="Development smoke only; benchmark uses native task steps")
    parser.add_argument("--wall-budget-s", type=float, help="Development smoke only; no total wall deadline in benchmark")
    parser.add_argument("--prompt")
    parser.add_argument("--exit-after-turn", action=argparse.BooleanOptionalAction)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, help="Use the native public task instruction; score and export actual video after the actor stops, then release resources")
    parser.add_argument("--visual-review-model", choices=["gpt-6.1-sol"], help="Optional counted RGB second-opinion model")
    parser.add_argument("--no-visual-review-model", action="store_const", const=None, dest="visual_review_model", help="Disable the optional visual helper")
    parser.add_argument("--pi05", action=argparse.BooleanOptionalAction, help="Enable matched ARX X5 pi0.5 short-prefix tool; AURIGA_PI05_SERVER_URL reuses a running service")
    parser.add_argument("--actor-payload-profile", choices=ACTOR_PAYLOAD_PROFILES, help="Model-facing JSON presentation; programs and immutable evidence retain full data")
    parser.add_argument("--enable-wait", action=argparse.BooleanOptionalAction,
                        help="Enable bounded native waiting with chronological RGB samples (default: enabled; --no-enable-wait disables)")
    parser.add_argument("--tracking-guard", action=argparse.BooleanOptionalAction, help="Stop EE motion after sustained measured joint tracking divergence; no automatic correction")
    parser.add_argument("--arm-queue", action=argparse.BooleanOptionalAction, help="Expose bounded per-arm stage queues")
    parser.add_argument("--motion-cost-preview", action=argparse.BooleanOptionalAction, help="Expose bounded planning-only candidate cost comparison")
    parser.add_argument("--bounded-continuation", action=argparse.BooleanOptionalAction, help="Bound repeated final-answer continuations without native progress")
    parser.add_argument("--motion-safety-guard", action=argparse.BooleanOptionalAction, help="Reject abnormal winding or over-budget compiled segments before dispatch")
    parser.add_argument("--trajectory-preview", action=argparse.BooleanOptionalAction, help="Disclose candidate joints and calibrated robot geometry for actor path review")
    parser.add_argument("--manipulation-protocol", action=argparse.BooleanOptionalAction, help="Use explicit contact, transfer and placement evidence in planning and bounded templates")
    for name in ('task-scheduling', 'operation-candidates', 'preview-intent', 'recovery-evidence'):
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction)
    parser.add_argument('--action-trajectory-delivery', choices=('automatic', 'diagnostic'),
                        help='Automatic sampled joints and images after actions, or compact routine summaries with full diagnostics on exceptions')
    parser.add_argument("--airborne-handoff", action=argparse.BooleanOptionalAction,
                        help="Permit direct airborne handoff; --no-airborne-handoff enforces surface-only transfer. Does not disable dual-arm motion")
    parser.add_argument("--handoff-preference", choices=("surface", "airborne"),
                        help="Soft planning preference when airborne handoff is allowed; new runs prefer a stable physical surface while leaving the choice to the actor")
    parser.add_argument("--joint-replan", action=argparse.BooleanOptionalAction, help="Try native planning to a closer equivalent joint endpoint and retain only a validated shorter path")
    parser.add_argument("--motion-execution-mode", choices=("resampled", "feedback"),
                        help="Use resampled unified actions (default); feedback is for trusted diagnostics and cannot run benchmarks")
    parser.add_argument("--default-motion-speed-scale", type=float,
                        help="Arm command spacing when a motion request omits speed selection (0.25..2; default 1)")
    parser.add_argument("--completion-guidance", choices=COMPLETION_GUIDANCE_MODES,
                        help="Benchmark completion guidance: state-aware hook by default; initial/both add an initial reminder; none disables the home check. Hook is dormant outside benchmark.")
    parser.add_argument("--default-task-notes", action=argparse.BooleanOptionalAction,
                        help="Include versioned operation experience in new benchmark prompts (default: enabled)")
    parser.add_argument("--task-notes", help="Additional public guidance appended to default operation notes; never hidden thresholds, object truth or scripted actions")
    parser.add_argument("--task-notes-source", help="Public source for --task-notes; saved with the actual prompt")
    for name in ('external-perception', 'vggt-height', 'object-table', 'scene3d',
                 'object-table-observe', 'scene3d-observe', 'obs3d-action-overlay', 'pose-skill', 'held-perception'):
        parser.add_argument('--' + name, action=argparse.BooleanOptionalAction)
    parser.add_argument('--vggt-height-mode', choices=('optional', 'default'))
    parser.add_argument('--vggt-endpoint')
    parser.add_argument('--vggt-support-z-m', type=float)
    parser.add_argument('--vggt-timeout-s', type=float)
    parser.add_argument('--object-table-max-objects', type=int)
    parser.add_argument('--object-table-max-height-m', type=float)
    parser.add_argument('--object-table-head-exclude-px')
    parser.add_argument('--scene3d-voxel-m', type=float)
    parser.add_argument('--scene3d-z-above-m', type=float)
    parser.add_argument('--scene3d-z-below-m', type=float)
    parser.add_argument('--obs3d-timeout-s', type=float)
    parser.add_argument('--obs3d-max-objects', type=int)
    parsed = vars(parser.parse_args())
    try:
        config = Config.load(parsed.get('config'), overrides=cli_overrides(parsed)).validate_launch()
        unknown_tools = set(config.features.disabled_tools) - create_registry(_configured_tool_specs(config), runtime=config).names - CONFIG_TOOL_NAMES
        if unknown_tools:
            raise ConfigError('unknown_disabled_tools:' + ','.join(sorted(unknown_tools)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if parsed.get('print_config'):
        print(json.dumps(config.to_dict(), indent=2))
        return
    args = argparse.Namespace(**namespace_values(config), resolved_config=config)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
