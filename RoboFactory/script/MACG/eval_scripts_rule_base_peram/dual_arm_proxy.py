#!/usr/bin/env python3
"""HTTP proxy that combines two per-arm VLA servers into one 16-DoF policy."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import pickle
from typing import Any

import numpy as np
import requests


LOGGER = logging.getLogger("dual_arm_proxy")
DEFAULT_ARM0_URL = "http://localhost:20011"
DEFAULT_ARM1_URL = "http://localhost:20012"
REQUIRED_HORIZON = 50


class ProxyError(RuntimeError):
    """Raised when an upstream request or response is invalid."""


def _encode_pickle(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value)).decode("utf-8")


def _decode_pickle(value: str) -> Any:
    return pickle.loads(base64.b64decode(value))


def build_arm_example(example: dict[str, Any], arm_id: int) -> dict[str, Any]:
    """Select only the observation fields consumed by one per-arm server."""
    if arm_id not in (0, 1):
        raise ValueError(f"arm_id must be 0 or 1, got {arm_id}")

    arm_keys = (
        f"image_agent{arm_id}",
        f"joint_position_agent{arm_id}",
        f"gripper_position_agent{arm_id}",
    )
    required_keys = ("global_image", *arm_keys, "task")
    missing = [key for key in required_keys if key not in example]
    if missing:
        raise ProxyError(f"arm{arm_id} request is missing fields: {missing}")

    return {key: example[key] for key in required_keys}


class UpstreamClient:
    def __init__(self, arm_name: str, server_url: str, timeout_s: float = 60.0):
        self.arm_name = arm_name
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            payload = response.json()
            healthy = (
                payload.get("status") == "healthy"
                and payload.get("model_loaded", True) is not False
            )
            return {
                "ok": healthy,
                "url": self.server_url,
                "response": payload,
            }
        except Exception as exc:
            return {
                "ok": False,
                "url": self.server_url,
                "error": str(exc),
            }

    def infer(self, example: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.server_url}/infer",
                json={"example": _encode_pickle(example)},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise ProxyError(
                f"{self.arm_name} request to {self.server_url} failed: {exc}"
            ) from exc

        if not payload.get("success"):
            raise ProxyError(
                f"{self.arm_name} server {self.server_url} returned an error: "
                f"{payload.get('error', 'unknown error')}"
            )
        if "result" not in payload:
            raise ProxyError(
                f"{self.arm_name} server {self.server_url} response has no result"
            )

        try:
            result = _decode_pickle(payload["result"])
        except Exception as exc:
            raise ProxyError(
                f"{self.arm_name} server {self.server_url} returned invalid result data: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise ProxyError(
                f"{self.arm_name} server {self.server_url} result must be a dict"
            )
        return result


class DualArmProxy:
    def __init__(
        self,
        arm0_url: str,
        arm1_url: str,
        timeout_s: float = 60.0,
        client_factory=UpstreamClient,
    ):
        self.clients = (
            client_factory("arm0", arm0_url, timeout_s),
            client_factory("arm1", arm1_url, timeout_s),
        )

    def health(self) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(client.health) for client in self.clients]
            arm_health = [future.result() for future in futures]
        return {
            "status": "healthy" if all(item["ok"] for item in arm_health) else "unhealthy",
            "model_loaded": all(item["ok"] for item in arm_health),
            "arms": {"arm0": arm_health[0], "arm1": arm_health[1]},
        }

    def infer_combined(self, example: dict[str, Any]) -> dict[str, np.ndarray]:
        arm_examples = (
            build_arm_example(example, 0),
            build_arm_example(example, 1),
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(client.infer, arm_example)
                for client, arm_example in zip(self.clients, arm_examples)
            ]
            results = [future.result() for future in futures]

        actions = []
        for arm_id, result in enumerate(results):
            if "actions" not in result:
                raise ProxyError(f"arm{arm_id} result has no actions")
            array = np.asarray(result["actions"])
            if array.ndim != 2 or array.shape[1] != 8:
                raise ProxyError(
                    f"arm{arm_id} actions must have shape [T, 8], got {array.shape}"
                )
            if array.shape[0] < REQUIRED_HORIZON:
                raise ProxyError(
                    f"arm{arm_id} action horizon must be at least "
                    f"{REQUIRED_HORIZON}, got {array.shape[0]}"
                )
            actions.append(array)

        if actions[0].shape[0] != actions[1].shape[0]:
            raise ProxyError(
                "arm action horizons differ: "
                f"arm0={actions[0].shape[0]}, arm1={actions[1].shape[0]}"
            )

        combined = np.concatenate(actions, axis=1)
        LOGGER.info(
            "Combined arm0 %s and arm1 %s into %s",
            actions[0].shape,
            actions[1].shape,
            combined.shape,
        )
        return {"actions": combined}


def make_handler(proxy: DualArmProxy):
    class ProxyRequestHandler(BaseHTTPRequestHandler):
        def _write_json(self, payload: dict[str, Any], status: int) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            payload = proxy.health()
            status = (
                HTTPStatus.OK
                if payload["status"] == "healthy"
                else HTTPStatus.SERVICE_UNAVAILABLE
            )
            self._write_json(payload, status)

        def do_POST(self) -> None:
            if self.path != "/infer":
                self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict) or "example" not in payload:
                    raise ProxyError("request JSON must contain an example field")
                example = _decode_pickle(payload["example"])
                if not isinstance(example, dict):
                    raise ProxyError("decoded example must be a dict")
                result = proxy.infer_combined(example)
                self._write_json(
                    {
                        "success": True,
                        "result": _encode_pickle(result),
                        "actions_shape": list(result["actions"].shape),
                    },
                    HTTPStatus.OK,
                )
            except ProxyError as exc:
                LOGGER.error("Inference failed: %s", exc)
                self._write_json(
                    {"success": False, "error": str(exc)},
                    HTTPStatus.BAD_GATEWAY,
                )
            except Exception as exc:
                LOGGER.exception("Unexpected inference failure")
                self._write_json(
                    {"success": False, "error": str(exc)},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, message_format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), message_format % args)

    return ProxyRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm0-url",
        default=os.getenv("ARM0_VLA_SERVER_URL", DEFAULT_ARM0_URL),
    )
    parser.add_argument(
        "--arm1-url",
        default=os.getenv("ARM1_VLA_SERVER_URL", DEFAULT_ARM1_URL),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("PERARM_PROXY_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PERARM_PROXY_PORT", "31888")),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.getenv("PERARM_INFER_TIMEOUT_S", "60")),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    proxy = DualArmProxy(args.arm0_url, args.arm1_url, args.timeout_s)
    LOGGER.info("arm0 upstream: %s", args.arm0_url)
    LOGGER.info("arm1 upstream: %s", args.arm1_url)
    LOGGER.info("local unified endpoint: http://%s:%s", args.host, args.port)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(proxy))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
