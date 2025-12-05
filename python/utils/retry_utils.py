# python/retry_utils.py
"""
Retry utilities with exponential backoff for HTTP requests.

This module provides decorators and functions to retry failed requests with configurable
exponential backoff, which is essential for handling transient network errors
and rate limiting in API calls.
"""

import functools
import logging
import random
import time
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

import requests

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.
    
    Attributes:
        max_retries: Maximum number of retry attempts (excluding the initial attempt)
        initial_delay: Initial delay before first retry in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff (e.g., 2.0 doubles delay each retry)
        jitter: If True, adds random jitter to delays to prevent thundering herd
        retriable_status_codes: HTTP status codes that should trigger a retry
        retriable_exceptions: Exception types that should trigger a retry
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 10.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        retriable_status_codes: Optional[Tuple[int, ...]] = None,
        retriable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
        # Default retriable status codes: rate limited, server errors
        self.retriable_status_codes = retriable_status_codes or (
            408,  # Request Timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        )
        
        # Default retriable exceptions
        self.retriable_exceptions = retriable_exceptions or (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        )
    
    @classmethod
    def default(cls) -> 'RetryConfig':
        """Create a default retry configuration."""
        return cls()
    
    @classmethod
    def aggressive(cls) -> 'RetryConfig':
        """Create a configuration for aggressive retries (good for HFT scenarios)."""
        return cls(
            max_retries=5,
            initial_delay=0.05,
            max_delay=5.0,
            backoff_multiplier=1.5,
        )
    
    @classmethod
    def conservative(cls) -> 'RetryConfig':
        """Create a configuration for conservative retries (good for rate-limited APIs)."""
        return cls(
            max_retries=3,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_multiplier=2.0,
        )
    
    def delay_for_attempt(self, attempt: int) -> float:
        """
        Calculate the delay for a given attempt number (0-indexed).
        
        Args:
            attempt: The attempt number (0 for first retry, 1 for second, etc.)
            
        Returns:
            Delay in seconds
        """
        if attempt < 0:
            return 0.0
        
        delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay
    
    def is_retriable_status(self, status_code: int) -> bool:
        """Check if an HTTP status code is retriable."""
        return status_code in self.retriable_status_codes
    
    def is_retriable_exception(self, exc: Exception) -> bool:
        """Check if an exception is retriable."""
        return isinstance(exc, self.retriable_exceptions)


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Decorator to add retry logic to a function that makes HTTP requests.
    
    The decorated function should raise exceptions or return a requests.Response
    object that can be checked for status codes.
    
    Args:
        config: Retry configuration (uses default if not provided)
        on_retry: Optional callback function called on each retry with (attempt, exception)
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @with_retry(RetryConfig.default())
        def fetch_data(url: str) -> dict:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """
    if config is None:
        config = RetryConfig.default()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result is a Response object with error status
                    if isinstance(result, requests.Response) and not result.ok:
                        if config.is_retriable_status(result.status_code):
                            if attempt < config.max_retries:
                                delay = config.delay_for_attempt(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{config.max_retries} for "
                                    f"HTTP {result.status_code}, waiting {delay:.2f}s"
                                )
                                if on_retry:
                                    on_retry(attempt, requests.HTTPError(response=result))
                                time.sleep(delay)
                                continue
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    if config.is_retriable_exception(e):
                        if attempt < config.max_retries:
                            delay = config.delay_for_attempt(attempt)
                            logger.debug(
                                f"Retry {attempt + 1}/{config.max_retries} for "
                                f"{type(e).__name__}: {e}, waiting {delay:.2f}s"
                            )
                            if on_retry:
                                on_retry(attempt, e)
                            time.sleep(delay)
                            continue
                    
                    # Check for HTTPError with retriable status
                    if isinstance(e, requests.HTTPError):
                        if e.response is not None and config.is_retriable_status(e.response.status_code):
                            if attempt < config.max_retries:
                                delay = config.delay_for_attempt(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{config.max_retries} for "
                                    f"HTTP {e.response.status_code}, waiting {delay:.2f}s"
                                )
                                if on_retry:
                                    on_retry(attempt, e)
                                time.sleep(delay)
                                continue
                    
                    # Non-retriable exception
                    raise
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            
            raise RuntimeError("Retry logic error: no result or exception")
        
        return wrapper
    
    return decorator


def retry_request(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs,
) -> T:
    """
    Execute a function with retry logic.
    
    This is a functional alternative to the @with_retry decorator.
    
    Args:
        func: Function to execute
        config: Retry configuration (uses default if not provided)
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Result of the function call
    
    Example:
        result = retry_request(
            lambda: requests.get(url).json(),
            config=RetryConfig.conservative()
        )
    """
    if config is None:
        config = RetryConfig.default()
    
    @with_retry(config)
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped()


def make_retriable_request(
    session: requests.Session,
    method: str,
    url: str,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> requests.Response:
    """
    Make an HTTP request with automatic retry on failure.
    
    Args:
        session: requests.Session to use
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        config: Retry configuration (uses default if not provided)
        **kwargs: Additional arguments to pass to session.request()
    
    Returns:
        requests.Response object
    
    Raises:
        requests.HTTPError: If all retries fail with error status
        requests.RequestException: If all retries fail with connection error
    
    Example:
        session = requests.Session()
        response = make_retriable_request(
            session, 'GET', 'https://api.example.com/data',
            config=RetryConfig.default()
        )
    """
    if config is None:
        config = RetryConfig.default()
    
    last_exception: Optional[Exception] = None
    last_response: Optional[requests.Response] = None
    
    for attempt in range(config.max_retries + 1):
        try:
            response = session.request(method, url, **kwargs)
            
            if response.ok:
                return response
            
            last_response = response
            
            if config.is_retriable_status(response.status_code):
                if attempt < config.max_retries:
                    delay = config.delay_for_attempt(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{config.max_retries} for "
                        f"HTTP {response.status_code} from {url}, waiting {delay:.2f}s"
                    )
                    time.sleep(delay)
                    continue
            
            # Non-retriable status code
            response.raise_for_status()
            return response  # Should not reach here
            
        except requests.HTTPError:
            raise
        except Exception as e:
            last_exception = e
            
            if config.is_retriable_exception(e):
                if attempt < config.max_retries:
                    delay = config.delay_for_attempt(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{config.max_retries} for "
                        f"{type(e).__name__} from {url}, waiting {delay:.2f}s"
                    )
                    time.sleep(delay)
                    continue
            
            raise
    
    # All retries exhausted
    if last_response is not None:
        last_response.raise_for_status()
    if last_exception is not None:
        raise last_exception
    
    raise RuntimeError("Retry logic error: no result or exception")
