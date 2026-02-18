"""Environment variable utilities."""
import os


def getvar(name: str) -> str:
    """
    Get an environment variable value.
    
    Args:
        name: Name of the environment variable
        
    Returns:
        str: Value of the environment variable
        
    Raises:
        ValueError: If the environment variable is not set
    """
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value
