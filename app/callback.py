from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class CompetitionCallback:
    def __init__(
        self,
        url: str | None,
        timeout_seconds: float = 10.0,
        attempts: int = 3,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts

    def send_success(
        self,
        request_id: str,
        evaluation_id: str,
        output_dir: Path,
    ) -> None:
        if not self.url:
            return
        body = json.dumps(
            {
                "request_id": request_id,
                "evaluationId": evaluation_id,
                "predPath": str(output_dir),
            },
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    if 200 <= response.status < 300:
                        return
                    raise OSError(f"callback returned HTTP {response.status}")
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(2**attempt)
        raise OSError(f"callback failed after {self.attempts} attempts: {last_error}")

