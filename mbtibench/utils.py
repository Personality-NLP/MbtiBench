import os
from typing import Optional, Tuple

from dotenv import load_dotenv


def get_base_url_and_api_key(host: Optional[str], port: Optional[int]) -> Tuple[str, str]:
    if host is None and port is None:
        load_dotenv()
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("API_KEY")
    elif host is not None and port is not None:
        base_url = f"http://{host}:{port}/v1"
        api_key = "EMPTY"
    else:
        raise ValueError("Either both host and port should be provided or neither")

    return base_url, api_key
