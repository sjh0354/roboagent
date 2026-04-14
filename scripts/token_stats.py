import os
import time
from datetime import datetime

import pandas as pd
import requests

BASE_URL = "https://open.xiaojingai.com/api/log/self"

COOKIE = os.getenv(
    "XIAOJINGAI_COOKIE",
    "Hm_lvt_1d94d83931290ca9998d455a8c8d3830=1774684215; HMACCOUNT=00D4FC1A3FDD4104; Hm_lvt_ae2900368485f7be1ebd264b352fc19c=1774684215; Hm_lvt_f666bba23fe08bff853f06dfae8fb3c6=1774870194; HMACCOUNT=00D4FC1A3FDD4104; Hm_lpvt_f666bba23fe08bff853f06dfae8fb3c6=1774931698; session=MTc3NTU2MzgxNHxEWDhFQVFMX2dBQUJFQUVRQUFEX2tmLUFBQVVHYzNSeWFXNW5EQVlBQkhKdmJHVURhVzUwQkFJQUFnWnpkSEpwYm1jTUNBQUdjM1JoZEhWekEybHVkQVFDQUFJR2MzUnlhVzVuREFjQUJXZHliM1Z3Qm5OMGNtbHVad3dKQUFka1pXWmhkV3gwQm5OMGNtbHVad3dFQUFKcFpBTnBiblFFQkFELVFNSUdjM1J5YVc1bkRBb0FDSFZ6WlhKdVlXMWxCbk4wY21sdVp3d0lBQVpyWVhwMWMyRT18QdoUHwC-yzjoJioks5KiAuC2IT3YM6Hm2gvOenT03Uw=; Hm_lpvt_1d94d83931290ca9998d455a8c8d3830=1775563904; Hm_lpvt_ae2900368485f7be1ebd264b352fc19c=1775563904",
)
NEW_API_USER = os.getenv("XIAOJINGAI_NEW_API_USER", "")

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "referer": "https://open.xiaojingai.com/console/log",
    "user-agent": "Mozilla/5.0",
    "cookie": COOKIE,
    "New-Api-User": NEW_API_USER,
}

def build_http_error(resp: requests.Response) -> RuntimeError:
    body = resp.text.strip().replace("\n", " ")
    if len(body) > 500:
        body = f"{body[:500]}..."
    return RuntimeError(
        f"HTTP {resp.status_code} for {resp.url}\n"
        f"response headers: {dict(resp.headers)}\n"
        f"response body: {body or '<empty>'}"
    )

def to_ts(s: str) -> int:
    # 例如 "2026-04-03 00:00:00"
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())

def fetch_page(
    page: int,
    page_size: int = 100,
    log_type: int = 0,
    token_name: str = "",
    model_name: str = "",
    start_timestamp: int = 1775145600,
    end_timestamp: int = 1775198445,
    group: str = "",
    request_id: str = "",
):
    params = {
        "p": page,
        "page_size": page_size,
        "type": log_type,
        "token_name": token_name,
        "model_name": model_name,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "group": group,
        "request_id": request_id,
    }

    resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    if not resp.ok:
        raise build_http_error(resp)

    data = resp.json()
    if not data.get("success", False):
        raise RuntimeError(f"API returned failure: {data}")

    return data

def ts_to_str(ts: int):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def fetch_all(
    page_size: int = 100,
    log_type: int = 0,
    token_name: str = "",
    model_name: str = "",
    start_timestamp: int = 1775145600,
    end_timestamp: int = 1775198445,
    group: str = "",
    request_id: str = "",
    sleep_sec: float = 0.2,
):
    first = fetch_page(
        page=1,
        page_size=page_size,
        log_type=log_type,
        token_name=token_name,
        model_name=model_name,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        group=group,
        request_id=request_id,
    )

    total = first["data"]["total"]
    items = list(first["data"]["items"])
    print(f"total = {total}, first page = {len(items)}")

    total_pages = (total + page_size - 1) // page_size

    for page in range(2, total_pages + 1):
        data = fetch_page(
            page=page,
            page_size=page_size,
            log_type=log_type,
            token_name=token_name,
            model_name=model_name,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            group=group,
            request_id=request_id,
        )
        page_items = data["data"]["items"]
        items.extend(page_items)
        print(f"page {page}/{total_pages}: +{len(page_items)}")
        time.sleep(sleep_sec)

    # 展平并加上可读时间
    rows = []
    for x in items:
        row = dict(x)
        row["created_at_str"] = ts_to_str(x["created_at"])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = fetch_all(
        page_size=100,
        log_type=0,
        token_name="",
        model_name="",
        start_timestamp = to_ts("2026-04-08 17:40:00"),
        end_timestamp = to_ts("2026-04-08 18:07:00"),
        group="",
        request_id="",
    )

    # 保存原始数据
    df.to_csv("/Users/kazusa/Documents/RAS_interactivate_planner/experiments/xiaojingai_usage_logs.csv", index=False, encoding="utf-8-sig")
    print("saved: xiaojingai_usage_logs.csv")

    # 简单统计
    if df.empty:
        summary = pd.DataFrame(
            columns=[
                "token_name",
                "model_name",
                "calls",
                "total_quota",
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_use_time",
            ]
        )
    else:
        summary = (
            df.groupby(["token_name", "model_name"], dropna=False)
            .agg(
                calls=("request_id", "count"),
                total_quota=("quota", "sum"),
                total_prompt_tokens=("prompt_tokens", "sum"),
                total_completion_tokens=("completion_tokens", "sum"),
                total_use_time=("use_time", "sum"),
            )
            .reset_index()
            .sort_values("total_quota", ascending=False)
        )

    summary.to_csv("/Users/kazusa/Documents/RAS_interactivate_planner/experiments/xiaojingai_usage_summary.csv", index=False, encoding="utf-8-sig")
    print("saved: xiaojingai_usage_summary.csv")
    print(summary.head(20))
