import functools
import json
import os
from datetime import timezone

from loguru import logger


def serialize(record):
    timestamp_utc = record["time"].astimezone(tz=timezone.utc)
    subset = {"timestamp": str(timestamp_utc)}
    return json.dumps(subset)


def sink(message):
    serialized = serialize(message.record)
    print(serialized)


fmt = "[{time}] - [{level}] - {module}:{message} {extra}"
config = {
    "handlers": [
        {
            "sink": sink,
            "format": fmt,
            "level": os.environ.get("ENV_VAR_LOG_LEVEL", "DEBUG"),
        },
    ],
}
logger.configure(**config)  # type: ignore


def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.log(
                    level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs
                )
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper
