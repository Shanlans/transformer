#!/usr/bin/env python3
"""
Unified Logging System

This module provides a structured logging system for the transformer project.
It replaces scattered print statements with organized, configurable logging.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from .log_config import LogConfig, get_env_config


class LogLevel(Enum):
    """Log levels for different types of messages."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Categories for organizing log messages."""
    SYSTEM = "SYSTEM"
    EXPERIMENT = "EXPERIMENT"
    TRAINING = "TRAINING"
    CLOUD = "CLOUD"
    CHECKPOINT = "CHECKPOINT"
    CONFIG = "CONFIG"
    CLEANUP = "CLEANUP"
    VALIDATION = "VALIDATION"


class UnifiedLogger:
    """
    Unified logging system for the transformer project.
    
    Features:
    - Structured log levels and categories
    - Configurable verbosity
    - Consistent formatting
    - Progress tracking
    - Error handling
    """
    
    def __init__(self, verbose: bool = True, log_file: Optional[str] = None, quiet: bool = False):
        """
        Initialize the unified logger.
        
        Args:
            verbose: Whether to show detailed output
            log_file: Optional log file path
            quiet: Whether to suppress most output
        """
        # Get configuration from environment
        env_config = get_env_config()
        
        self.verbose = verbose and not quiet
        self.quiet = quiet
        self.log_file = log_file or env_config.get('log_file')
        self.start_time = datetime.now()
        
        # Create log directory if needed
        if self.log_file:
            LogConfig.create_log_directory()
        
        # Setup logging
        self._setup_logging()
        
        # Progress tracking
        self.current_operation = None
        self.operation_start_time = None
        
    def _setup_logging(self):
        """Setup logging configuration."""
        # Determine log level
        if self.quiet:
            level = logging.WARNING
        elif self.verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        # Create formatter
        if self.quiet:
            # Minimal format for quiet mode
            formatter = logging.Formatter('%(message)s')
        else:
            # Full format for normal/verbose mode
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        # Setup file handler if specified
        handlers = [console_handler]
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True
        )
        
        self.logger = logging.getLogger('transformer')
    
    def _format_message(self, level: LogLevel, category: LogCategory, message: str) -> str:
        """Format log message with level and category."""
        icons = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.SUCCESS: "✅",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🚨"
        }
        
        icon = icons.get(level, "📝")
        return f"{icon} [{category.value}] {message}"
    
    def log(self, level: LogLevel, category: LogCategory, message: str, **kwargs):
        """Log a message with specified level and category."""
        formatted_message = self._format_message(level, category, message)
        
        if level == LogLevel.DEBUG:
            self.logger.debug(formatted_message, **kwargs)
        elif level == LogLevel.INFO:
            self.logger.info(formatted_message, **kwargs)
        elif level == LogLevel.SUCCESS:
            self.logger.info(formatted_message, **kwargs)
        elif level == LogLevel.WARNING:
            self.logger.warning(formatted_message, **kwargs)
        elif level == LogLevel.ERROR:
            self.logger.error(formatted_message, **kwargs)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(formatted_message, **kwargs)
    
    def debug(self, category: LogCategory, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, message, **kwargs)
    
    def info(self, category: LogCategory, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, category, message, **kwargs)
    
    def success(self, category: LogCategory, message: str, **kwargs):
        """Log success message."""
        self.log(LogLevel.SUCCESS, category, message, **kwargs)
    
    def warning(self, category: LogCategory, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, category, message, **kwargs)
    
    def error(self, category: LogCategory, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, category, message, **kwargs)
    
    def critical(self, category: LogCategory, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, category, message, **kwargs)
    
    def start_operation(self, operation: str, category: LogCategory = LogCategory.SYSTEM):
        """Start tracking an operation."""
        self.current_operation = operation
        self.operation_start_time = datetime.now()
        self.info(category, f"Starting {operation}")
    
    def end_operation(self, success: bool = True, category: LogCategory = LogCategory.SYSTEM):
        """End tracking an operation."""
        if self.current_operation and self.operation_start_time:
            duration = (datetime.now() - self.operation_start_time).total_seconds()
            status = "completed" if success else "failed"
            level = LogLevel.SUCCESS if success else LogLevel.ERROR
            self.log(level, category, f"{self.current_operation} {status} in {duration:.2f}s")
            
            self.current_operation = None
            self.operation_start_time = None
    
    def print_header(self, title: str, width: int = 80):
        """Print a formatted header."""
        if self.verbose:
            print("\n" + "=" * width)
            print(f" {title}")
            print("=" * width)
    
    def print_section(self, title: str, width: int = 60):
        """Print a formatted section header."""
        if self.verbose:
            print(f"\n{'-' * width}")
            print(f" {title}")
            print(f"{'-' * width}")
    
    def print_table(self, headers: List[str], rows: List[List[str]], title: Optional[str] = None):
        """Print a formatted table."""
        if not self.verbose:
            return
            
        if title:
            self.print_section(title)
        
        if not rows:
            self.info(LogCategory.SYSTEM, "No data to display")
            return
        
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print header
        header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in rows:
            row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            print(row_str)
    
    def print_progress(self, current: int, total: int, operation: str = "Progress"):
        """Print progress information."""
        if not self.verbose:
            return
            
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 30
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        print(f"\r{operation}: |{bar}| {percentage:.1f}% ({current}/{total})", end="", flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def print_summary(self, title: str, data: Dict[str, Any]):
        """Print a summary with key-value pairs."""
        if not self.verbose:
            return
            
        self.print_section(title)
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    def set_verbose(self, verbose: bool):
        """Set verbosity level."""
        self.verbose = verbose
        self._setup_logging()


# Global logger instance
_global_logger = None


def get_logger(verbose: bool = True, log_file: Optional[str] = None, quiet: bool = False) -> UnifiedLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger(verbose=verbose, log_file=log_file, quiet=quiet)
    return _global_logger


def set_verbose(verbose: bool):
    """Set global verbosity level."""
    logger = get_logger()
    logger.set_verbose(verbose)


def set_quiet(quiet: bool):
    """Set global quiet mode."""
    logger = get_logger()
    logger.quiet = quiet
    logger.verbose = logger.verbose and not quiet
    logger._setup_logging()


# Convenience functions for common logging patterns
def log_system(message: str, level: LogLevel = LogLevel.INFO):
    """Log system message."""
    get_logger().log(level, LogCategory.SYSTEM, message)


def log_experiment(message: str, level: LogLevel = LogLevel.INFO):
    """Log experiment message."""
    get_logger().log(level, LogCategory.EXPERIMENT, message)


def log_training(message: str, level: LogLevel = LogLevel.INFO):
    """Log training message."""
    get_logger().log(level, LogCategory.TRAINING, message)


def log_cloud(message: str, level: LogLevel = LogLevel.INFO):
    """Log cloud message."""
    get_logger().log(level, LogCategory.CLOUD, message)


def log_checkpoint(message: str, level: LogLevel = LogLevel.INFO):
    """Log checkpoint message."""
    get_logger().log(level, LogCategory.CHECKPOINT, message)


def log_config(message: str, level: LogLevel = LogLevel.INFO):
    """Log config message."""
    get_logger().log(level, LogCategory.CONFIG, message)


def log_cleanup(message: str, level: LogLevel = LogLevel.INFO):
    """Log cleanup message."""
    get_logger().log(level, LogCategory.CLEANUP, message)


def log_validation(message: str, level: LogLevel = LogLevel.INFO):
    """Log validation message."""
    get_logger().log(level, LogCategory.VALIDATION, message)


# Context manager for operations
class OperationContext:
    """Context manager for tracking operations."""
    
    def __init__(self, operation: str, category: LogCategory = LogCategory.SYSTEM):
        self.operation = operation
        self.category = category
        self.logger = get_logger()
    
    def __enter__(self):
        self.logger.start_operation(self.operation, self.category)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        self.logger.end_operation(success, self.category)
        return False  # Don't suppress exceptions


# Decorator for operation tracking
def track_operation(operation: str, category: LogCategory = LogCategory.SYSTEM):
    """Decorator to track operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with OperationContext(operation, category):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the logger
    logger = get_logger(verbose=True)
    
    logger.print_header("Logger Test")
    
    logger.info(LogCategory.SYSTEM, "Testing info message")
    logger.success(LogCategory.EXPERIMENT, "Testing success message")
    logger.warning(LogCategory.TRAINING, "Testing warning message")
    logger.error(LogCategory.CLOUD, "Testing error message")
    
    logger.print_table(
        ["Name", "Value", "Status"],
        [
            ["Item 1", "100", "Active"],
            ["Item 2", "200", "Inactive"],
            ["Item 3", "300", "Pending"]
        ],
        "Test Table"
    )
    
    logger.print_summary("Test Summary", {
        "Total Items": 3,
        "Active": 1,
        "Inactive": 1,
        "Pending": 1
    })
    
    with OperationContext("Test Operation"):
        import time
        time.sleep(0.1)
    
    logger.success(LogCategory.SYSTEM, "Logger test completed")
