"""Small plugin ports; native backends and raw persistence stay with their owners."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from ...core.contracts import digest


@dataclass(frozen=True)
class ToolAvailability:
    features: object
    disabled_tools: frozenset
    observation_profile: str
    capabilities: dict | None
    enable_wait: bool
    vision_compare: bool
    visual_review: bool
    policy: bool
    programs: bool
    delegate_policy: bool = False

    @classmethod
    def from_runtime(cls, runtime):
        return cls(runtime.config.features, frozenset(runtime.disabled_tools),
            runtime.observation_profile, deepcopy(runtime.capabilities), runtime.enable_wait,
            runtime.vision_compare, runtime.visual_reviewer is not None,
            runtime.action_policy is not None, runtime.programs is not None,
            getattr(runtime, 'delegate_policy', None) is not None)


@dataclass(frozen=True)
class ToolContext:
    identity: dict
    observation: dict | None
    capabilities: dict | None
    action_budget: dict
    read_action: Callable
    execute: Callable | None = None

    @classmethod
    def from_runtime(cls, runtime, name, effect, correlation):
        correlation = deepcopy(correlation)
        sequence = 0
        def execute(action):
            nonlocal sequence
            # A repeated tool invocation owns the same ordered action IDs.
            # Harness then enforces immutable content, admission and no replay
            # of an uncertain acknowledgement, exactly as for built-in actions.
            if not correlation.get('tool_call_id'):
                raise ValueError('plugin_physical_action_requires_tool_call_id')
            identifier = 'action-plugin-' + digest([name, correlation, sequence])[:24]
            sequence += 1
            action = {**action, 'action_id': identifier}
            return runtime.execute(action, correlation)
        return cls(deepcopy(runtime.identity), deepcopy(runtime.latest),
            deepcopy(runtime.capabilities), deepcopy(runtime.action_budget()),
            lambda identifier: deepcopy(runtime.action(identifier)),
            execute if effect == 'physical' else None)
