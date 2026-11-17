"""Claude Code agent backend (optional extra ``auriga-harness[claude]``; design rev 3).

Importing this package never imports ``claude_agent_sdk``: ``options`` and ``projector`` load it lazily, so a checkout
without the extra can still import ``auriga.cortex.agents``.
"""
