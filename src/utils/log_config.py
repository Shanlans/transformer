#!/usr/bin/env python3
"""
Logging Configuration

This module provides configuration for the unified logging system.
"""

import os
from enum import Enum


class LogConfig:
    """Configuration for logging system."""
    
    # Default log levels
    DEFAULT_LEVEL = "INFO"
    QUIET_LEVEL = "WARNING"
    VERBOSE_LEVEL = "DEBUG"
    
    # Log categories and their default levels
    CATEGORY_LEVELS = {
        "SYSTEM": "INFO",
        "EXPERIMENT": "INFO", 
        "TRAINING": "INFO",
        "CLOUD": "INFO",
        "CHECKPOINT": "INFO",
        "CONFIG": "INFO",
        "CLEANUP": "INFO",
        "VALIDATION": "INFO"
    }
    
    # Environment-based configuration
    @staticmethod
    def get_log_level(verbose: bool = True, quiet: bool = False) -> str:
        """Get log level based on verbosity settings."""
        if quiet:
            return LogConfig.QUIET_LEVEL
        elif verbose:
            return LogConfig.VERBOSE_LEVEL
        else:
            return LogConfig.DEFAULT_LEVEL
    
    @staticmethod
    def should_log(category: str, level: str, verbose: bool = True, quiet: bool = False) -> bool:
        """Determine if a log message should be displayed."""
        if quiet:
            return level in ["ERROR", "CRITICAL"]
        elif verbose:
            return True
        else:
            return level in ["INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    
    @staticmethod
    def get_log_file() -> str:
        """Get log file path."""
        return os.path.join("logs", "transformer.log")
    
    @staticmethod
    def create_log_directory():
        """Create log directory if it doesn't exist."""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)


# Environment variables for configuration
def get_env_config():
    """Get configuration from environment variables."""
    return {
        "verbose": os.getenv("TRANSFORMER_VERBOSE", "true").lower() == "true",
        "quiet": os.getenv("TRANSFORMER_QUIET", "false").lower() == "true",
        "log_file": os.getenv("TRANSFORMER_LOG_FILE", LogConfig.get_log_file()),
        "log_level": os.getenv("TRANSFORMER_LOG_LEVEL", LogConfig.DEFAULT_LEVEL)
    }
