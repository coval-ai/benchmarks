"""The LiveKit variant of the voice-orchestration benchmark agent.

Vapi, Telnyx and Retell host the agent behind a REST object the runner patches;
LiveKit has none, so this package *is* the agent. It runs as its own image
(``Dockerfile.livekit``) on LiveKit Cloud and needs the ``livekit`` extra.
"""
