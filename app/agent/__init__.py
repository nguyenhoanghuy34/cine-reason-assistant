__all__ = ["agent"]

try:
    from app.agent.graph import agent
except Exception:  # pragma: no cover - allow non-graph imports during tests
    agent = None