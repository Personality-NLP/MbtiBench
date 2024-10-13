import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


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


def get_credit_info():
    base_url, api_key = get_base_url_and_api_key(None, None)

    now_date = datetime.now()
    one_month_ago = now_date - relativedelta(months=1)
    now_date = now_date.strftime("%Y-%m-%d")
    one_month_ago = one_month_ago.strftime("%Y-%m-%d")

    url_usage = f"{base_url}/dashboard/billing/usage?start_date={one_month_ago}&end_date={now_date}"
    url_subscription = f"{base_url}/dashboard/billing/subscription"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    conn = requests.get(url_subscription, headers=headers)
    if conn.status_code == 200:
        subscriptionData = conn.json()
        totalAmount = subscriptionData["hard_limit_usd"]

    conn = requests.get(url_usage, headers=headers)
    if conn.status_code == 200:
        usageData = conn.json()
        totalUsage = usageData["total_usage"] / 100

    remaining = totalAmount - totalUsage

    logger.info(f"Total:  {totalAmount:.2f}$")
    logger.info(f"Usage:  {totalUsage:.2f}$")
    logger.info(f"Remain: {remaining:.2f}$")
