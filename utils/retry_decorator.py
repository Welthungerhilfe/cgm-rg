import anyio
import functools
import logging

def retry(retries=3, delay=2):
    """Decorator to retry an async function with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)  # Call the function
                except Exception as e:
                    if attempt < retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logging.warning(f"Retrying {func.__name__} (Attempt {attempt+1}/{retries}) in {wait_time}s due to: {e}")
                        await anyio.sleep(wait_time)
                    else:
                        logging.error(f"{func.__name__} failed after {retries} attempts: {e}")
                        raise e  # Raise the last exception if retries fail
        return wrapper
    return decorator
