# tests/test_retry_utils.py
"""
Tests for the retry utilities module.
"""

import time
import unittest
from unittest.mock import Mock, patch

import requests
import requests.exceptions

from python.retry_utils import (
    RetryConfig,
    with_retry,
    retry_request,
    make_retriable_request,
)


class TestRetryConfig(unittest.TestCase):
    """Tests for RetryConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig.default()
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.initial_delay, 0.1)
        self.assertEqual(config.max_delay, 10.0)
        self.assertEqual(config.backoff_multiplier, 2.0)
        self.assertTrue(config.jitter)

    def test_aggressive_config(self):
        """Test aggressive configuration values."""
        config = RetryConfig.aggressive()
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.initial_delay, 0.05)
        self.assertEqual(config.max_delay, 5.0)
        self.assertEqual(config.backoff_multiplier, 1.5)

    def test_conservative_config(self):
        """Test conservative configuration values."""
        config = RetryConfig.conservative()
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.initial_delay, 0.5)
        self.assertEqual(config.max_delay, 30.0)
        self.assertEqual(config.backoff_multiplier, 2.0)

    def test_delay_calculation_without_jitter(self):
        """Test delay calculation without jitter."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.1,
            max_delay=10.0,
            backoff_multiplier=2.0,
            jitter=False,
        )
        
        self.assertAlmostEqual(config.delay_for_attempt(0), 0.1, places=5)
        self.assertAlmostEqual(config.delay_for_attempt(1), 0.2, places=5)
        self.assertAlmostEqual(config.delay_for_attempt(2), 0.4, places=5)
        self.assertAlmostEqual(config.delay_for_attempt(3), 0.8, places=5)

    def test_delay_capped_at_max(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            max_retries=10,
            initial_delay=1.0,
            max_delay=5.0,
            backoff_multiplier=2.0,
            jitter=False,
        )
        
        # After several attempts, delay should be capped at max_delay
        self.assertEqual(config.delay_for_attempt(5), 5.0)
        self.assertEqual(config.delay_for_attempt(10), 5.0)

    def test_is_retriable_status(self):
        """Test retriable status code detection."""
        config = RetryConfig.default()
        
        # Retriable status codes
        self.assertTrue(config.is_retriable_status(408))  # Request Timeout
        self.assertTrue(config.is_retriable_status(429))  # Too Many Requests
        self.assertTrue(config.is_retriable_status(500))  # Internal Server Error
        self.assertTrue(config.is_retriable_status(502))  # Bad Gateway
        self.assertTrue(config.is_retriable_status(503))  # Service Unavailable
        self.assertTrue(config.is_retriable_status(504))  # Gateway Timeout
        
        # Non-retriable status codes
        self.assertFalse(config.is_retriable_status(200))
        self.assertFalse(config.is_retriable_status(400))
        self.assertFalse(config.is_retriable_status(401))
        self.assertFalse(config.is_retriable_status(403))
        self.assertFalse(config.is_retriable_status(404))

    def test_is_retriable_exception(self):
        """Test retriable exception detection."""
        config = RetryConfig.default()
        
        # Retriable exceptions
        self.assertTrue(config.is_retriable_exception(requests.exceptions.ConnectionError()))
        self.assertTrue(config.is_retriable_exception(requests.exceptions.Timeout()))
        self.assertTrue(config.is_retriable_exception(requests.exceptions.ChunkedEncodingError()))
        
        # Non-retriable exceptions
        self.assertFalse(config.is_retriable_exception(ValueError()))
        self.assertFalse(config.is_retriable_exception(KeyError()))


class TestWithRetryDecorator(unittest.TestCase):
    """Tests for with_retry decorator."""

    def test_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        config = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False)
        call_count = 0
        
        @with_retry(config)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = succeed()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    def test_success_after_retries(self):
        """Test success after several failures."""
        config = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False)
        call_count = 0
        
        @with_retry(config)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.ConnectionError("Temporary failure")
            return "success"
        
        result = fail_then_succeed()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    def test_non_retriable_exception_no_retry(self):
        """Test that non-retriable exceptions are not retried."""
        config = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False)
        call_count = 0
        
        @with_retry(config)
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retriable")
        
        with self.assertRaises(ValueError):
            raise_value_error()
        
        # Should fail immediately without retries
        self.assertEqual(call_count, 1)

    def test_retries_exhausted(self):
        """Test that exception is raised after all retries are exhausted."""
        config = RetryConfig(max_retries=2, initial_delay=0.001, jitter=False)
        call_count = 0
        
        @with_retry(config)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.ConnectionError("Persistent failure")
        
        with self.assertRaises(requests.exceptions.ConnectionError):
            always_fail()
        
        # Initial attempt + 2 retries = 3 total attempts
        self.assertEqual(call_count, 3)


class TestRetryRequest(unittest.TestCase):
    """Tests for retry_request function."""

    def test_retry_request_success(self):
        """Test successful request with retry_request."""
        config = RetryConfig(max_retries=3, initial_delay=0.001, jitter=False)
        
        result = retry_request(
            lambda: "success",
            config=config,
        )
        self.assertEqual(result, "success")


class TestMakeRetriableRequest(unittest.TestCase):
    """Tests for make_retriable_request function."""

    @patch('python.retry_utils.time.sleep')
    def test_success_on_first_attempt(self, mock_sleep):
        """Test successful request on first attempt."""
        session = Mock(spec=requests.Session)
        response = Mock(spec=requests.Response)
        response.ok = True
        response.status_code = 200
        session.request.return_value = response
        
        config = RetryConfig(max_retries=3, initial_delay=0.1, jitter=False)
        
        result = make_retriable_request(session, 'GET', 'http://example.com', config=config)
        
        self.assertEqual(result, response)
        session.request.assert_called_once_with('GET', 'http://example.com')
        mock_sleep.assert_not_called()

    @patch('python.retry_utils.time.sleep')
    def test_retry_on_server_error(self, mock_sleep):
        """Test retry on server error (5xx)."""
        session = Mock(spec=requests.Session)
        
        # First two calls return 503, third succeeds
        error_response = Mock(spec=requests.Response)
        error_response.ok = False
        error_response.status_code = 503
        error_response.raise_for_status.side_effect = requests.HTTPError(response=error_response)
        
        success_response = Mock(spec=requests.Response)
        success_response.ok = True
        success_response.status_code = 200
        
        session.request.side_effect = [error_response, error_response, success_response]
        
        config = RetryConfig(max_retries=3, initial_delay=0.1, jitter=False)
        
        result = make_retriable_request(session, 'GET', 'http://example.com', config=config)
        
        self.assertEqual(result, success_response)
        self.assertEqual(session.request.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Two retries

    @patch('python.retry_utils.time.sleep')
    def test_retry_on_connection_error(self, mock_sleep):
        """Test retry on connection error."""
        session = Mock(spec=requests.Session)
        
        success_response = Mock(spec=requests.Response)
        success_response.ok = True
        success_response.status_code = 200
        
        # First call raises ConnectionError, second succeeds
        session.request.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            success_response,
        ]
        
        config = RetryConfig(max_retries=3, initial_delay=0.1, jitter=False)
        
        result = make_retriable_request(session, 'GET', 'http://example.com', config=config)
        
        self.assertEqual(result, success_response)
        self.assertEqual(session.request.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch('python.retry_utils.time.sleep')
    def test_no_retry_on_client_error(self, mock_sleep):
        """Test that client errors (4xx, except 408/429) are not retried."""
        session = Mock(spec=requests.Session)
        
        error_response = Mock(spec=requests.Response)
        error_response.ok = False
        error_response.status_code = 404
        error_response.raise_for_status.side_effect = requests.HTTPError(response=error_response)
        
        session.request.return_value = error_response
        
        config = RetryConfig(max_retries=3, initial_delay=0.1, jitter=False)
        
        with self.assertRaises(requests.HTTPError):
            make_retriable_request(session, 'GET', 'http://example.com', config=config)
        
        # Should fail immediately without retries
        session.request.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
