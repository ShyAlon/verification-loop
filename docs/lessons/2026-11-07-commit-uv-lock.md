# 2026-11-07 - Commit the uv lockfile

uv.lock was untracked here while delivery-api and interview-engine track
theirs (CI caches on it). An untracked lockfile drifts silently per machine.
Committed; `uv sync --locked` now verifies the environment against git.
