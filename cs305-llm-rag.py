
from __future__ import annotations
import json
from datetime import date
from typing import Callable
import requests
from config import get_settings

settings = get_settings()

class AgentClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {settings.api_token}"})

    def generate(
        self, payload: dict, on_response: Callable[[str], None] | None = None
    ) -> str:
        payload = payload.copy()
        payload["prompt"] = (
            f"Current date: {date.today().isoformat()}\n\n{payload['prompt']}"
        )
        try:
            with self.session.post(
                f"{settings.server_url.rstrip('/')}/generate",
                json=payload,
                timeout=settings.request_timeout_seconds,
                stream=True,
            ) as response:
                response.raise_for_status()
                full_response = ""
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    if "token" in event:
                        full_response += event["token"]
                        if on_response:
                            on_response(full_response)
                return full_response.strip()
        except requests.Timeout as error:
            raise RuntimeError("The Kaggle model took too long to respond.") from error
        except RuntimeError:
            raise
        except json.JSONDecodeError as error:
            raise RuntimeError("Kaggle returned an invalid streaming response.") from error
        except requests.HTTPError as error:
            detail = error.response.text[:1000] if error.response is not None else str(error)
            raise RuntimeError(f"Kaggle model request failed: {detail}") from error
        except requests.RequestException as error:
            raise RuntimeError(f"Could not reach the Kaggle model: {error}") from error


def create_client() -> AgentClient:
    return AgentClient()


def generate_xai_requirements(
    user_requirement: str,
    client: AgentClient,
    history: list[dict[str, str]] | None = None,
    on_response: Callable[[str], None] | None = None,
) -> str:
    return client.generate(
        {"mode": "requirements", "prompt": user_requirement, "history": history or []},
        on_response=on_response,
    )


def generate_sdlc_recommendation(
    requirements_srs: str,
    user_question: str,
    client: AgentClient,
    history: list[dict[str, str]] | None = None,
    on_response: Callable[[str], None] | None = None,
) -> str:
    return client.generate(
        {
            "mode": "sdlc",
            "prompt": user_question,
            "requirements_srs": requirements_srs,
            "history": history or [],
        },
        on_response=on_response,
    )


def verify_requirements(
    requirements_srs: str,
    user_question: str,
    client: AgentClient,
    history: list[dict[str, str]] | None = None,
    on_response: Callable[[str], None] | None = None,
) -> str:
    return client.generate(
        {
            "mode": "requirements_verifier",
            "prompt": user_question,
            "artifact_to_verify": requirements_srs,
            "history": history or [],
        },
        on_response=on_response,
    )


def verify_sdlc_recommendation(
    sdlc_recommendation: str,
    user_question: str,
    client: AgentClient,
    history: list[dict[str, str]] | None = None,
    on_response: Callable[[str], None] | None = None,
) -> str:
    return client.generate(
        {
            "mode": "sdlc_verifier",
            "prompt": user_question,
            "artifact_to_verify": sdlc_recommendation,
            "history": history or [],
        },
        on_response=on_response,
    )


def main() -> None:
    prompt = input("Describe the software scenario: ").strip()
    if prompt:
        print(generate_xai_requirements(prompt, create_client()))


if __name__ == "__main__":
    main()
