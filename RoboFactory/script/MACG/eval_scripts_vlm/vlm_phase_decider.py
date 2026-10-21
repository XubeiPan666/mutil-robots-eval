"""OpenRouter-backed high-level instruction scheduler for stack-bowl evaluation."""

from __future__ import annotations

import base64
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import requests
from PIL import Image


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _to_uint8_rgb(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        scale = 255.0 if array.size and float(array.max()) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return array


class VLMPhaseDecider:
    """Ask a VLM for a constrained phase or pair of atomic instructions.

    The class is deliberately independent of ManiSkill so it can be unit-tested
    with saved frames. Any API failure keeps the current phase and uses its known
    in-distribution instruction pair.
    """

    def __init__(
        self,
        *,
        api_key: str,
        phase_config: Mapping[str, Any],
        model: str = "qwen/qwen3.5-27b",
        mode: str = "phase",
        log_dir: str = "tmp/vlm_logs",
        seed: int = 0,
        no_gripper_numbers: bool = False,
        request_timeout_s: float = 60.0,
        max_retries: int = 2,
        max_total_wait_s: float = 90.0,
        min_steps_per_phase: Optional[int] = None,
        min_confidence: float = 0.6,
        image_upscale: int = 2,
        save_images: bool = True,
        api_url: str = OPENROUTER_URL,
        verbose: bool = False,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        if mode not in {"phase", "per_agent"}:
            raise ValueError(f"Unsupported VLM mode: {mode}")

        self.api_key = api_key
        self.model = model
        self.mode = mode
        self.api_url = api_url
        self.no_gripper_numbers = no_gripper_numbers
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.max_total_wait_s = max_total_wait_s
        self.min_confidence = min_confidence
        self.image_upscale = max(1, int(image_upscale))
        self.save_images = save_images
        self.verbose = verbose

        self.phase_instructions = {
            int(key): dict(value)
            for key, value in phase_config["phase_instructions"].items()
        }
        self.phase_semantics = {
            int(key): str(value)
            for key, value in phase_config["phase_semantics"].items()
        }
        self.task_brief = str(phase_config["task_brief"])
        self.atomic_actions = list(phase_config["atomic_actions"])
        self.gripper_threshold = float(
            phase_config.get("gripper_threshold", 0.02)
        )
        self.min_steps_per_phase = int(
            min_steps_per_phase
            if min_steps_per_phase is not None
            else phase_config.get("min_steps_per_phase", 2)
        )
        self.current_phase = int(phase_config.get("initial_phase", 1))
        self.steps_in_phase = 0

        run_name = f"seed_{seed}_{mode}_{int(time.time())}"
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.log_dir / "images"
        if self.save_images:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "decisions.jsonl"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/mas-vla/RoboFactory",
                "X-Title": "RoboFactory VLM Phase Scheduler",
            }
        )

    def _system_prompt(self) -> str:
        phases = "\n".join(
            f"Phase {phase}: {meaning}"
            for phase, meaning in sorted(self.phase_semantics.items())
        )
        actions = "\n".join(f'- "{action}"' for action in self.atomic_actions)
        return (
            "You are the high-level scheduler for a dual-Panda robot cell.\n"
            f"Task: {self.task_brief}\n"
            "Images are ordered and labeled: dedicated global VLM camera, "
            "agent0 wrist camera, agent1 wrist camera.\n"
            f"Gripper opening semantics: below {self.gripper_threshold:.3f} m "
            "means CLOSED/holding; otherwise OPEN.\n\n"
            f"{phases}\n\n"
            "State-machine rules:\n"
            "- Output only the current phase or the next phase; never skip or go back.\n"
            "- Keep the current phase until its stated completion condition is visible "
            "and consistent with the gripper state.\n"
            "- A closed gripper alone does not prove an object is held; inspect images.\n"
            "- If uncertain, keep the current phase.\n\n"
            "Allowed per-agent instructions:\n"
            f"{actions}"
        )

    def _schema(self, allowed_phases: Sequence[int]) -> Dict[str, Any]:
        properties: Dict[str, Any] = {
            "observed": {
                "type": "string",
                "description": "One short sentence describing both arms and bowls.",
            },
            "phase": {"type": "integer", "enum": list(allowed_phases)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }
        required = ["observed", "phase", "confidence"]
        if self.mode == "per_agent":
            instruction_schema = {"type": "string", "enum": self.atomic_actions}
            properties["actions"] = {
                "type": "object",
                "properties": {
                    "agent0": instruction_schema,
                    "agent1": instruction_schema,
                },
                "required": ["agent0", "agent1"],
                "additionalProperties": False,
            }
            required.append("actions")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "stack_bowl_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def _encode_image(
        self, image: Any, label: str, step: int
    ) -> tuple[Dict[str, Any], Optional[str]]:
        array = _to_uint8_rgb(image)
        pil_image = Image.fromarray(array, mode="RGB")
        if self.image_upscale > 1 and min(pil_image.size) <= 224:
            pil_image = pil_image.resize(
                (
                    pil_image.width * self.image_upscale,
                    pil_image.height * self.image_upscale,
                ),
                Image.Resampling.LANCZOS,
            )
        path: Optional[Path] = None
        if self.save_images:
            path = self.image_dir / f"step_{step:04d}_{label}.jpg"
            pil_image.save(path, "JPEG", quality=90)
        buffer = io.BytesIO()
        pil_image.save(buffer, "JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return (
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            },
            str(path) if path is not None else None,
        )

    def _fallback(self, reason: str) -> Dict[str, Any]:
        return {
            "observed": reason,
            "phase": self.current_phase,
            "confidence": 0.0,
            "actions": dict(self.phase_instructions[self.current_phase]),
            "fallback": True,
        }

    def _write_log(self, record: Mapping[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def decide(
        self,
        *,
        images: Mapping[str, Any],
        agent0_gripper: float,
        agent1_gripper: float,
        step: int,
        shadow_phase: Optional[int] = None,
        elapsed_policy_steps: int = 1,
    ) -> Dict[str, Any]:
        self.steps_in_phase += max(1, int(elapsed_policy_steps))
        next_phase = min(self.current_phase + 1, max(self.phase_instructions))
        allowed_phases = sorted({self.current_phase, next_phase})

        state_lines = [
            f"current_phase={self.current_phase}",
            f"steps_in_phase={self.steps_in_phase}",
        ]
        if not self.no_gripper_numbers:
            state_lines.extend(
                [
                    f"agent0_gripper={agent0_gripper:.5f} "
                    f"({'CLOSED' if agent0_gripper < self.gripper_threshold else 'OPEN'})",
                    f"agent1_gripper={agent1_gripper:.5f} "
                    f"({'CLOSED' if agent1_gripper < self.gripper_threshold else 'OPEN'})",
                ]
            )
        else:
            state_lines.append(
                "Gripper numbers are intentionally hidden; infer state from images."
            )
        state_lines.append(
            f"Choose phase from {allowed_phases}. "
            "Describe evidence briefly, then make the constrained decision."
        )

        content = [{"type": "text", "text": "\n".join(state_lines)}]
        image_paths: Dict[str, Optional[str]] = {}
        for label in ("global", "agent0_wrist", "agent1_wrist"):
            if label not in images:
                raise KeyError(f"Missing required VLM image: {label}")
            encoded, path = self._encode_image(images[label], label, step)
            content.append({"type": "text", "text": f"Image: {label}"})
            content.append(encoded)
            image_paths[label] = path

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 220,
            "reasoning": {"enabled": False},
            "response_format": self._schema(allowed_phases),
        }

        started = time.monotonic()
        deadline = started + self.max_total_wait_s
        response_body: Dict[str, Any] = {}
        error: Optional[str] = None
        decision: Optional[Dict[str, Any]] = None
        retries = 0
        for attempt in range(self.max_retries + 1):
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                error = (
                    f"VLM decision exceeded {self.max_total_wait_s:.1f}s "
                    "total wait budget"
                )
                break
            try:
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    timeout=min(self.request_timeout_s, remaining_s),
                )
                response_body = response.json()
                if response.status_code == 200 and response_body.get("choices"):
                    raw = response_body["choices"][0]["message"].get("content")
                    decision = json.loads(raw)
                    break
                error = (
                    f"HTTP {response.status_code}: "
                    f"{response_body.get('error', response_body)}"
                )
                retryable = response.status_code == 429 or response.status_code >= 500
            except (
                requests.RequestException,
                TypeError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"
                retryable = True
            if not retryable or attempt >= self.max_retries:
                break
            retries += 1
            delay = min(20.0, 2.0**attempt) + random.uniform(0, 0.5)
            if self.verbose:
                print(f"[VLM] {error}; retrying in {delay:.1f}s")
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))

        latency_s = time.monotonic() - started
        if decision is None:
            decision = self._fallback(error or "empty VLM response")
        else:
            valid_decision = True
            try:
                proposed_phase = int(decision["phase"])
                confidence = float(decision["confidence"])
                if self.mode == "per_agent":
                    actions = decision["actions"]
                    if set(actions) != {"agent0", "agent1"}:
                        raise ValueError("actions must contain exactly both agents")
                    if any(
                        action not in self.atomic_actions
                        for action in actions.values()
                    ):
                        raise ValueError("response contains an unknown action")
            except (KeyError, TypeError, ValueError) as exc:
                valid_decision = False
                decision = self._fallback(
                    f"invalid structured response: {type(exc).__name__}: {exc}"
                )
                proposed_phase = self.current_phase
                confidence = 0.0
            if valid_decision:
                if (
                    proposed_phase not in allowed_phases
                    or (
                        proposed_phase > self.current_phase
                        and self.steps_in_phase < self.min_steps_per_phase
                    )
                    or (
                        proposed_phase > self.current_phase
                        and confidence < self.min_confidence
                    )
                    or (
                        self.mode == "per_agent"
                        and confidence < self.min_confidence
                    )
                ):
                    decision = self._fallback(
                        f"guard rejected phase {proposed_phase} "
                        f"at confidence {confidence:.3f}"
                    )
                else:
                    if proposed_phase != self.current_phase:
                        self.current_phase = proposed_phase
                        self.steps_in_phase = 0
                    if self.mode == "phase":
                        decision["actions"] = dict(
                            self.phase_instructions[self.current_phase]
                        )
                    decision["fallback"] = False

        decision["phase"] = self.current_phase
        decision["latency_s"] = latency_s
        decision["retries"] = retries
        decision["shadow_phase"] = shadow_phase

        usage = response_body.get("usage", {})
        record = {
            "timestamp": time.time(),
            "step": step,
            "model": self.model,
            "mode": self.mode,
            "current_phase": self.current_phase,
            "steps_in_phase": self.steps_in_phase,
            "shadow_phase": shadow_phase,
            "shadow_agreement": (
                self.current_phase == shadow_phase
                if shadow_phase is not None
                else None
            ),
            "agent0_gripper": (
                None if self.no_gripper_numbers else agent0_gripper
            ),
            "agent1_gripper": (
                None if self.no_gripper_numbers else agent1_gripper
            ),
            "image_paths": image_paths,
            "decision": decision,
            "raw_response": response_body,
            "usage": usage,
            "error": error,
        }
        self._write_log(record)
        if self.verbose:
            print(
                f"[VLM] step={step} phase={self.current_phase} "
                f"shadow={shadow_phase} latency={latency_s:.2f}s "
                f"actions={decision['actions']}"
            )
        return decision

    def instruction_prompt(self, decision: Mapping[str, Any]) -> str:
        actions = decision["actions"]
        return f"{actions['agent0']} | {actions['agent1']}"
