import logging
import sys

# Basic configuration
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Get the root logger
logger = logging.getLogger() # Get root logger
logger.setLevel(LOGGING_LEVEL)

# # Remove existing handlers to avoid duplicate logs if this is called multiple times
# for handler in logger.handlers[:]:
#     logger.removeHandler(handler)

# Create handler (console)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(LOGGING_LEVEL)

# Create formatter
formatter = logging.Formatter(LOGGING_FORMAT)

# Add formatter to handler
handler.setFormatter(formatter)

# Add handler to the root logger
# Check if handler already exists to prevent duplicates during potential reloads
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger instance."""
    return logging.getLogger(name)

# # Example usage:
# import logging
# logger = logging.getLogger(__name__)
# logger.info("info message")
# logger.error("error message")
