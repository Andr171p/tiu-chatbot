from typing import Any

import logging

from ddgs import DDGS
from langchain.tools import tool
from pydantic import BaseModel, Field

from . import browser, rag

INDEX_NAME = "knowledge-index"

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Входные аргументы для поиска информации"""

    query: str = Field(description="Запрос для поиска")


@tool(
    "web_search",
    description="""\
    Выполняет поиск в интернете.
    Возвращает список найденных страниц с заголовками, URL и кратким описанием.
    Подходит для получения актуальной информации из интернета.
    Используй этот инструмент экономно.
    """,
    args_schema=SearchInput
)
def web_search(query: str) -> list[dict[str, Any]]:
    logger.info("Searching in web for query - '%s ...'", query[:100])
    return DDGS().text(query, region="ru-ru", max_results=10)


class BrowsePageInput(BaseModel):
    """Входные аргументы для открытия страницы с помощью браузера"""

    link: str = Field(description="URL адрес страницы")


@tool(
    "browse_page",
    description="Открывает WEB-страницу и получает её контент в формате Markdown",
    args_schema=BrowsePageInput,
)
async def browse_page(link: str) -> str:
    return await browser.get_page_text(link)


@tool(
    "search_knowledge",
    description="Выполняет поиск по базе знаний приёмной комиссии ТИУ",
    args_schema=SearchInput
)
async def search_knowledge(query: str) -> str:
    docs = await rag.retrieve_documents(index_name=INDEX_NAME, query=query)
    return "\n\n".join(docs)
