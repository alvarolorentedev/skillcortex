from __future__ import annotations

import json
from typing import Any


class OpenAICompatApp:
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        try:
            if method == "GET" and path == "/healthz":
                return json_response(start_response, 200, {"status": "ok", "runtime": self.runtime.bundle.name})
            if method == "GET" and path == "/v1/models":
                return json_response(
                    start_response,
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": self.runtime.bundle.name,
                                "object": "model",
                                "owned_by": "slmcortex",
                            }
                        ],
                    },
                )
            if method == "POST" and path == "/v1/chat/completions":
                body = read_request_body(environ)
                payload = json.loads(body or "{}")
                return json_response(start_response, 200, self.runtime.chat_completion(payload))
            return json_response(start_response, 404, {"error": {"message": "not found"}})
        except ValueError as error:
            return json_response(start_response, 400, {"error": {"message": str(error)}})
        except Exception as error:  # pragma: no cover - defensive server boundary
            return json_response(start_response, 500, {"error": {"message": str(error)}})


def json_response(start_response, status_code: int, payload: dict[str, Any]):
    body = json.dumps(payload).encode() + b"\n"
    start_response(
        f"{status_code} {status_text(status_code)}",
        [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def read_request_body(environ) -> str:
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if length <= 0:
        return ""
    return environ["wsgi.input"].read(length).decode()


def status_text(status_code: int) -> str:
    return {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        500: "Internal Server Error",
    }[status_code]
