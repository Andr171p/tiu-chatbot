import asyncio
import logging
from uuid import uuid4

from src.agent import call_chatbot


async def main() -> None:
    user_id = uuid4().hex
    while True:
        user_prompt = input("[User] ")
        response = await call_chatbot(user_id, user_prompt)
        print(f"[AI] {response}")  # noqa: T201


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
