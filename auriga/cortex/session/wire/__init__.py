"""Model-relay wire adapters (format-specific half of ``ModelRequestProxy``).

Only stdlib, ``..usage`` and the stdlib-only ``auriga.cortex.agents.context_signal`` are
imported here, so the ``--http-worker`` subprocess can load an adapter without the rest of the
Harness.
"""
from .base import CODEX_PROFILE, OPENCODE_PROFILE, ClientProfile, Route, WireAdapter, WireReply
from .openai_responses import OpenAIResponsesWire, present_responses_request
from .anthropic_messages import AnthropicMessagesWire, present_messages_request

__all__ = ["AnthropicMessagesWire", "CODEX_PROFILE", "ClientProfile", "OPENCODE_PROFILE", "Route", "WireAdapter",
           "WireReply", "OpenAIResponsesWire", "present_messages_request", "present_responses_request",
           "wire_from_name"]


def wire_from_name(name, *, client=None):
    """Adapter instance for a recorded wire ``name`` and client profile ``client``.

    Used by the ``--http-worker`` subprocess, which receives both in its JSON payload.
    ``"openai_responses"`` accepts ``client`` None (= ``"codex"``, the payload of every Codex
    relay) or a supported profile (``"codex"``, ``"opencode"``). Every other wire needs its profile named explicitly, because
    the profile is part of the wire's identity: ``wire_from_name("anthropic_messages")`` without
    ``client`` raises ``ValueError("unknown_model_wire:anthropic_messages:client_profile_required")``.
    Unknown names raise ``ValueError("unknown_model_wire:<name>")``; unsupported profiles raise
    ``ValueError("unsupported_wire_client:<name>:<client>")``.
    """
    if name == OpenAIResponsesWire.name:
        return OpenAIResponsesWire(client=client)
    if name == AnthropicMessagesWire.name:
        if client is None:
            raise ValueError("unknown_model_wire:" + name + ":client_profile_required")
        return AnthropicMessagesWire(client=client)
    raise ValueError("unknown_model_wire:" + str(name))
