import socket
import time
import random
from functools import wraps
import ee
from google.auth.exceptions import TransportError
from googleapiclient.errors import HttpError
import httplib2
import urllib3
import requests

from src.utils.config import Config

logger = Config.get_logger()

# Standard transient network and API exceptions to retry on
DEFAULT_RETRY_EXCEPTIONS = (
    socket.gaierror,
    socket.timeout,
    TimeoutError,
    ConnectionError,
    TransportError,
    httplib2.ServerNotFoundError,
    httplib2.HttpLib2Error,
    HttpError,
    ee.EEException,
    urllib3.exceptions.HTTPError,
    requests.exceptions.RequestException,
)

def is_transient_error(e):
    """Helper to check if an exception is transient (should be retried).
    
    Returns False for permanent client/auth/permission errors.
    """
    if isinstance(e, HttpError):
        status = None
        if hasattr(e, 'resp') and e.resp is not None:
            status = getattr(e.resp, 'status', None)
        if status is None:
            status = getattr(e, 'status_code', None)
            
        if status is not None:
            try:
                status = int(status)
            except (ValueError, TypeError):
                pass
                
            if status in (400, 401, 404):
                return False
            if status == 403:
                # 403 Forbidden can be transient (rate limits, quota) or permanent (permissions)
                err_msg = str(e).lower()
                if "rate" in err_msg or "quota" in err_msg or "limit" in err_msg or "speed" in err_msg:
                    return True
                return False
            if 400 <= status < 500:
                # Other client errors are usually permanent
                return False
                
    elif isinstance(e, requests.exceptions.HTTPError):
        if e.response is not None:
            status = e.response.status_code
            if status in (400, 401, 404):
                return False
            if status == 403:
                err_msg = str(e).lower()
                if "rate" in err_msg or "quota" in err_msg or "limit" in err_msg or "speed" in err_msg:
                    return True
                return False
            if 400 <= status < 500:
                return False
                
    return True

def retry_on_network_error(max_retries=8, initial_delay=2.0, backoff_factor=2.0, jitter=True):
    """Decorator to retry a function if it raises transient network or API exceptions.
    
    Args:
        max_retries (int): Maximum number of retries before raising the exception.
        initial_delay (float): Delay in seconds before the first retry.
        backoff_factor (float): Multiplier for the delay on subsequent retries.
        jitter (bool): If True, adds a random jitter to the delay to prevent thundering herd.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except DEFAULT_RETRY_EXCEPTIONS as e:
                    # Check if this is a permanent error
                    if not is_transient_error(e):
                        logger.warning(
                            f"Permanent API/network error in '{func.__name__}': {e}. Bypassing retries."
                        )
                        raise
                        
                    # If we reached the maximum number of retries, raise the error.
                    if attempt == max_retries:
                        logger.error(
                            f"Function '{func.__name__}' failed after {max_retries} attempts due to: {e}"
                        )
                        raise
                        
                    # Calculate next delay with optional jitter
                    current_delay = delay
                    if jitter:
                        current_delay = delay * (0.5 + random.random())
                        
                    logger.warning(
                        f"Transient network or API error in '{func.__name__}' on attempt {attempt}/{max_retries}: {e}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    delay *= backoff_factor
        return wrapper
    return decorator

@retry_on_network_error()
def execute_with_retry(request):
    """Executes a Google API request (e.g. from service.files()) with transient error retries."""
    return request.execute()

@retry_on_network_error()
def get_info_with_retry(ee_object):
    """Fetches Earth Engine info synchronously with transient error retries."""
    return ee_object.getInfo()

@retry_on_network_error()
def get_task_status_with_retry(t_id):
    """Fetches Earth Engine task status with transient error retries."""
    return ee.data.getTaskStatus(t_id)[0]

@retry_on_network_error()
def start_task_with_retry(task):
    """Starts an Earth Engine task with transient error retries.
    
    Gracefully handles duplicate task start exceptions (in case a start request
    succeeded on the server but timed out client-side).
    """
    try:
        task.start()
    except ee.EEException as e:
        err_msg = str(e).lower()
        if "already exists" in err_msg or "already running" in err_msg or "has already started" in err_msg:
            logger.info(f"Task {task.description} was already started/exists. Proceeding.")
        else:
            raise
