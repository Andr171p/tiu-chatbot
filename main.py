import logging

import uvicorn
from fastapi import Body, FastAPI, status

from src.agent import call_chatbot

app = FastAPI(
    title="AI чат-бот приёмной комиссии ТИУ",
    description="""\
    Отвечает на вопросы абитуриентов и их родителей, используя базу знаний приёмной комиссии,
    персонализирует ответы основываясь на долгосрочную память.
    """,
    version="0.1.0",
)


@app.post(
    path="/chatbot",
    status_code=status.HTTP_200_OK,
    summary="Получить ответ от чат-бота"
)
async def generate_response(
        user_id: str = Body(..., embed=True, description="ID пользователя"),
        user_prompt: str = Body(..., embed=True, description="Запрос пользователя")
) -> dict[str, str]:
    return {"text": await call_chatbot(user_id, user_prompt)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
