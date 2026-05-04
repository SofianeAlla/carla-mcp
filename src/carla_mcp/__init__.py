__version__ = "2.1.2"

# In stdio MCP mode, libcarla's C++ side prints lines like
#     INFO:  Found the required file in cache!  Carla/Maps/Nav/Town03.bin
# directly to fd 1. Stdout is reserved for the MCP JSON-RPC framing, so those
# lines corrupt the stream and the client sees "Unexpected token 'I'" parse
# errors. Redirect fd 1 to a log file at import time (before `import carla`
# can happen anywhere downstream), and re-bind sys.stdout to the original fd
# so Python and MCP still talk to the client cleanly.
import os
import sys
import tempfile

if os.environ.get("CARLA_MCP_TRANSPORT", "stdio") == "stdio":
    _saved_stdout_fd = os.dup(1)
    _log_path = os.environ.get(
        "CARLA_MCP_LOG", os.path.join(tempfile.gettempdir(), "carla-mcp.log")
    )
    _log_fd = os.open(
        _log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
    )
    os.dup2(_log_fd, 1)
    os.close(_log_fd)
    sys.stdout = os.fdopen(_saved_stdout_fd, "w", buffering=1, encoding="utf-8")
