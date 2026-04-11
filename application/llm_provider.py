"""
Shared LLM routing: load .env, prefer Gemini (GOOGLE_API_KEY / GEMINI_API_KEY),
then OpenAI (caller-supplied key), then Ollama at OLLAMA_HOST.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

_DOTENV_LOADED = False


def load_env() -> None:
    """Load project-root .env once."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env")
    except ImportError:
        pass
    _DOTENV_LOADED = True


def gemini_api_key() -> str:
    load_env()
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def gemini_model_name() -> str:
    load_env()
    return (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()


def ollama_base_url() -> str:
    load_env()
    return (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def ollama_fallback_model() -> str:
    load_env()
    return (os.getenv("OLLAMA_FALLBACK_MODEL") or "llama3.2:1b").strip()


def openai_track_can_run(session_openai_key: Optional[str] = None) -> bool:
    """True if Gemini env, session OpenAI key, or Ollama host is configured (last resort)."""
    load_env()
    if gemini_api_key():
        return True
    key = (session_openai_key or "").strip()
    if key.startswith("sk-"):
        return True
    return bool((os.getenv("OLLAMA_HOST") or "").strip())


def chat_completion(
    messages: List[Dict[str, Any]],
    *,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-3.5-turbo",
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> str:
    """
    Return assistant text. Order: Gemini (.env) -> OpenAI (session key) -> Ollama.
    """
    load_env()
    err_parts: List[str] = []

    if gemini_api_key():
        try:
            return _gemini_chat(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            err_parts.append(f"Gemini: {e}")

    key = (openai_api_key or "").strip()
    if key.startswith("sk-"):
        try:
            return _openai_chat(messages, key, openai_model, max_tokens, temperature)
        except Exception as e:
            err_parts.append(f"OpenAI: {e}")

    try:
        return _ollama_chat(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        err_parts.append(f"Ollama: {e}")

    return "Error: " + ("; ".join(err_parts) if err_parts else "No LLM backend available.")


def _gemini_chat(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    import google.generativeai as genai

    genai.configure(api_key=gemini_api_key())
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    non_system = [m for m in messages if m.get("role") != "system"]
    if not non_system:
        return "Error: No user message for Gemini."

    user_blob_parts: List[str] = []
    for m in non_system:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "assistant":
            user_blob_parts.append(f"Assistant:\n{content}")
        elif role == "tool":
            user_blob_parts.append(f"Tool result:\n{content}")
        else:
            user_blob_parts.append(content)
    user_blob = "\n\n".join(user_blob_parts)

    model = genai.GenerativeModel(
        gemini_model_name(),
        system_instruction=system_instruction,
    )
    response = model.generate_content(
        user_blob,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    text = getattr(response, "text", None)
    if text:
        return text
    if response.candidates:
        parts = []
        for c in response.candidates:
            if c.content and c.content.parts:
                for p in c.content.parts:
                    if hasattr(p, "text") and p.text:
                        parts.append(p.text)
        if parts:
            return "\n".join(parts)
    return "Error: Empty response from Gemini."


def _openai_chat(
    messages: List[Dict[str, Any]],
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _ollama_chat(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    base = ollama_base_url()
    r = requests.post(
        f"{base}/api/chat",
        json={
            "model": ollama_fallback_model(),
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    if content:
        return content
    return f"Error: Unexpected Ollama response: {data!r}"


def chat_completion_excessive_agency(
    prompt: str,
    *,
    openai_api_key: Optional[str],
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Single user-message chat used by excessive agency (no system prompt)."""
    return chat_completion(
        [{"role": "user", "content": prompt}],
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def run_pizza_price_tool_conversation(
    user_input: str,
    *,
    openai_api_key: Optional[str],
    price_tool_spec: Dict[str, Any],
    get_pizza_price: Callable[[str], str],
) -> str:
    """
    Insecure plugin demo: OpenAI-style tool call, or Gemini tools, or Ollama tools.
    """
    load_env()

    if gemini_api_key():
        try:
            return _gemini_pizza_tools(user_input, get_pizza_price)
        except Exception:
            pass  # fall through to OpenAI / Ollama

    key = (openai_api_key or "").strip()
    if key.startswith("sk-"):
        try:
            return _openai_pizza_tools(user_input, key, get_pizza_price, price_tool_spec)
        except Exception as e:
            pass

    try:
        return _ollama_pizza_tools(user_input, get_pizza_price, price_tool_spec)
    except Exception as e:
        return f"Error: {e}"


def _openai_pizza_tools(
    user_input: str,
    api_key: str,
    get_pizza_price: Callable[[str], str],
    price_tool_spec: Dict[str, Any],
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful pizza shop assistant that can provide prices for different pizza types. ",
            },
            {"role": "user", "content": user_input},
        ],
        tools=[{"type": "function", "function": price_tool_spec}],
        tool_choice="auto",
    )
    message = response.choices[0].message
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.type == "function" and tool_call.function.name == "get_pizza_price":
                arguments = json.loads(tool_call.function.arguments)
                pizza_type = arguments.get("pizza_type")
                if pizza_type:
                    price = get_pizza_price(pizza_type)
                    second = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful pizza shop assistant that can provide prices for different pizza types.",
                            },
                            {"role": "user", "content": user_input},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": "get_pizza_price",
                                            "arguments": tool_call.function.arguments,
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"The price for {pizza_type} pizza is {price}",
                            },
                        ],
                    )
                    return second.choices[0].message.content or ""
    return message.content or ""


def _gemini_pizza_tools(
    user_input: str,
    get_pizza_price: Callable[[str], str],
) -> str:
    import google.generativeai as genai

    def get_pizza_price_tool(pizza_type: str) -> str:
        """Get the price for a specific pizza type."""
        return get_pizza_price(pizza_type)

    genai.configure(api_key=gemini_api_key())
    model = genai.GenerativeModel(
        gemini_model_name(),
        tools=[get_pizza_price_tool],
        system_instruction="You are a helpful pizza shop assistant that can provide prices for different pizza types.",
    )
    try:
        chat = model.start_chat(enable_automatic_function_calling=True)
    except TypeError:
        chat = model.start_chat()
    response = chat.send_message(user_input)
    return response.text


def _ollama_pizza_tools(
    user_input: str,
    get_pizza_price: Callable[[str], str],
    price_tool_spec: Dict[str, Any],
) -> str:
    base = ollama_base_url()
    r = requests.post(
        f"{base}/api/chat",
        json={
            "model": ollama_fallback_model(),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful pizza shop assistant that can provide prices for different pizza types.",
                },
                {"role": "user", "content": user_input},
            ],
            "stream": False,
            "tools": [{"type": "function", "function": price_tool_spec}],
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function") or {}
            if fn.get("name") == "get_pizza_price":
                arguments = json.loads(fn.get("arguments") or "{}")
                pizza_type = arguments.get("pizza_type")
                if pizza_type:
                    price = get_pizza_price(pizza_type)
                    r2 = requests.post(
                        f"{base}/api/chat",
                        json={
                            "model": ollama_fallback_model(),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful pizza shop assistant that can provide prices for different pizza types.",
                                },
                                {"role": "user", "content": user_input},
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": tool_calls,
                                },
                                {
                                    "role": "tool",
                                    "content": f"The price for {pizza_type} pizza is {price}",
                                },
                            ],
                            "stream": False,
                        },
                        timeout=120,
                    )
                    r2.raise_for_status()
                    data2 = r2.json()
                    c2 = (data2.get("message") or {}).get("content") or ""
                    return c2
    return (msg.get("content") or "") or "Error: Ollama tool flow did not return content."
