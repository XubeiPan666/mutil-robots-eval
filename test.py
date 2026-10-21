"""Manual smoke check for a user-configured OpenAI-compatible server."""

import os

from openai import OpenAI


def main():
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gemma4:26b"),
        messages=[{"role": "user", "content": "hello"}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
