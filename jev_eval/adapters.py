"""Small configurable HTTP adapters; credentials are read from the environment."""

import importlib
import json
import math
import os
import urllib.request

from .protocol import PROMPT_PREFIX, InputError, decode_answer, keys_for, task_json, wire
from .native import BACKENDS

APIS = ("jev", "chat", "responses", "anthropic", "native", "plugin")


def structured_signal(value: object, fields: tuple[str, ...]) -> bool:
    """Find structured refusal and tool signals without interpreting prose."""
    if isinstance(value, dict):
        return any(value.get(field) for field in fields) or any(structured_signal(child, fields) for child in value.values())
    if isinstance(value, list):
        return any(structured_signal(child, fields) for child in value)
    return False


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise InputError("configuration must be a JSON object")
    if config.get("api") not in APIS or not isinstance(config.get("model"), str) or not config["model"]:
        raise InputError("config requires api and model")
    params = config.get("parameters", {})
    if not isinstance(params, dict) or set(params) & {"model", "input", "messages", "state", "questions", "stream"}:
        raise InputError("parameters cannot replace the task or model")
    timeout = config.get("timeout_seconds", 120)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise InputError("timeout_seconds must be a positive finite number")
    for field in ("endpoint_env", "api_key_env", "checkpoint_env"):
        if field in config and (not isinstance(config[field], str) or not config[field]):
            raise InputError("environment variable names must be nonempty strings")
    if config["api"] == "plugin" and not isinstance(config.get("callable"), str):
        raise InputError("plugin config requires callable='module:function'")
    if config["api"] == "native":
        if config.get("backend") not in BACKENDS:
            raise InputError("unknown native backend")
        if not isinstance(config.get("checkpoint", config.get("checkpoint_env")), str):
            raise InputError("native config requires checkpoint or checkpoint_env")
        if type(config.get("max_input_tokens", 16384)) is not int or config.get("max_input_tokens", 16384) < 1:
            raise InputError("max_input_tokens must be a positive integer")
        if not isinstance(config.get("device", "cuda:0"), str) or params:
            raise InputError("native config requires a device string and no generation parameters")
    aliases = config.get("accepted_model_names", [config["model"]])
    if not isinstance(aliases, list) or not aliases or any(not isinstance(x, str) for x in aliases):
        raise InputError("accepted_model_names must be a nonempty string list")


def payload_for(question: dict, order: list[int], config: dict) -> dict:
    task = task_json(question, order)
    api = config["api"]
    if api in ("jev", "native", "plugin"):
        payload = {"model": config["model"], **task}
    elif api == "responses":
        payload = {"model": config["model"], "input": PROMPT_PREFIX + wire(task), "stream": False}
    else:
        payload = {"model": config["model"], "messages": [
            {"role": "user", "content": PROMPT_PREFIX + wire(task)}], "stream": False}
    payload.update(config.get("parameters", {}))
    return payload


def parse_response(raw: dict, question: dict, config: dict) -> str:
    """Return a candidate key, rejecting incomplete or ambiguous selections."""
    if not isinstance(raw, dict):
        raise ValueError("response must be an object")
    api = config["api"]
    keys = keys_for(question)
    if api == "plugin":
        return decode_answer(wire(raw), keys, question["options"])
    if raw.get("error"):
        raise ValueError("API error in response")
    if raw.get("model") not in config.get("accepted_model_names", [config["model"]]):
        raise ValueError("unexpected response model")
    if api in ("jev", "native"):
        answers = raw.get("answers")
        if not isinstance(answers, dict) or set(answers) != {"q"}:
            raise ValueError("unexpected question IDs")
        answer = answers["q"]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("unexpected native answer type")
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(keys):
            raise ValueError("probability keys mismatch")
        values = list(probs.values())
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("invalid probabilities")
        mass = sum(values)
        if mass <= 0 or abs(mass - 1) > .0051 * len(keys) + 1e-6:
            raise ValueError("invalid probability mass")
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in keys or probs[choice] + .0101 < max(values):
            raise ValueError("invalid native choice")
        return choice
    if api == "anthropic":
        blocks = raw.get("content")
        if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
            raise ValueError("invalid content blocks")
        texts = [b.get("text") for b in blocks if b.get("type") == "text"]
        finish = raw.get("stop_reason")
        valid_finish = finish in ("end_turn", "stop_sequence")
    elif api == "responses":
        output = raw.get("output")
        if raw.get("error") or not isinstance(output, list) or any(not isinstance(x, dict) for x in output):
            raise ValueError("invalid Responses output")
        if any(x.get("type") not in ("reasoning", "message") for x in output):
            raise ValueError("unexpected Responses item")
        messages = [x for x in output if x.get("type") == "message"]
        if len(messages) != 1 or messages[0].get("role") != "assistant" or messages[0].get("status") != "completed":
            raise ValueError("expected one completed assistant message")
        blocks = messages[0].get("content")
        if not isinstance(blocks, list) or any(not isinstance(b, dict) or b.get("type") != "output_text" for b in blocks):
            raise ValueError("invalid Responses content")
        texts = [b.get("text") for b in blocks]
        valid_finish = raw.get("status") == "completed"
    else:
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("expected one chat choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("invalid chat message")
        if structured_signal(raw, ("refusal", "tool_calls", "function_call")):
            raise ValueError("response contains a refusal or tool call")
        texts = [message.get("content")]
        valid_finish = choices[0].get("finish_reason") == "stop"
    if not valid_finish or any(not isinstance(t, str) for t in texts):
        raise ValueError("response does not contain a completed selection")
    return decode_answer("".join(texts), keys, question["options"])


class Adapter:
    """Use one configured HTTP API, native model, or user-provided callable."""

    def __init__(self, config: dict) -> None:
        validate_config(config)
        self.config = config
        self.callable = None
        self.endpoint = None
        self.headers = {"Content-Type": "application/json"}
        if config["api"] == "native":
            from .native import build_native
            backend = build_native(config)
            self.callable = lambda payload: {"model": config["model"], "answers": backend(payload)}
            return
        if config["api"] == "plugin":
            module, name = config["callable"].split(":", 1)
            self.callable = getattr(importlib.import_module(module), name)
            return
        endpoint = config.get("endpoint")
        if config.get("endpoint_env"):
            endpoint = os.environ.get(config["endpoint_env"])
        if not isinstance(endpoint, str) or not endpoint.startswith(("https://", "http://")):
            raise InputError("configure the complete endpoint URL or endpoint_env")
        self.endpoint = endpoint
        if config.get("api_key_env"):
            secret = os.environ.get(config["api_key_env"])
            if not secret:
                raise InputError("configured API key environment variable is unset")
            if config["api"] == "anthropic":
                self.headers["x-api-key"] = secret
            else:
                self.headers["Authorization"] = "Bearer " + secret
        if config["api"] == "anthropic":
            self.headers["anthropic-version"] = config.get("anthropic_version", "2023-06-01")

    def select(self, question: dict, order: list[int]) -> str:
        """Return an original candidate key after one model request."""
        payload = payload_for(question, order, self.config)
        if self.callable is not None:
            raw = self.callable(payload)
        else:
            request = urllib.request.Request(self.endpoint, data=wire(payload).encode(), headers=self.headers, method="POST")
            with urllib.request.urlopen(request, timeout=self.config.get("timeout_seconds", 120)) as response:
                raw = json.load(response)
        return parse_response(raw, question, self.config)
