#!/usr/bin/env python
"""Main entry point for the Rooney Capital live trading worker."""

import logging
import sys
from pathlib import Path

from src.runner.live_worker import LiveWorker, load_runtime_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def main() -> int:
    """Load configuration and start the live worker."""
    try:
        # Load runtime configuration
        config = load_runtime_config()

        logger.info("Starting Rooney Capital live trading worker")
        logger.info("Symbols: %s", ", ".join(config.symbols))

        # Create and run the worker
        worker = LiveWorker(config)
        worker.run()

        return 0

    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")
        return 0
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
