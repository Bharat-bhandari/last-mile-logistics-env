# Shim for openenv validator — it expects server/app.py at the project root
# with a top-level `app`, a callable `main()`, and an `if __name__ == '__main__'` guard.
from last_mile_env.server.app import app  # noqa: F401
import last_mile_env.server.app as _real_app


def main():  # noqa: D401
    """Entry point — delegates to the real server's main()."""
    _real_app.main()


if __name__ == '__main__':
    main()
