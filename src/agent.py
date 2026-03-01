import logging
import os

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .memory import UserContext, remember, search_memory
from .prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from .settings import SQLITE_PATH, current_datetime, settings
from .tools import browse_page, search_knowledge, web_search

logger = logging.getLogger(__name__)

model = ChatOpenAI(
    api_key=settings.yandex_cloud_api_key,
    model=settings.qwen3_235b,
    base_url=settings.yandex_cloud_base_url,
    temperature=0.3,
)


summarization_middleware = SummarizationMiddleware(
    model=model,
    summary_prompt=SUMMARY_PROMPT,
    trigger=("tokens", 9000),
    keep=("messages", 30),
)


async def call_chatbot(user_id: str, user_prompt: str) -> str:
    """Вызов AI чат-бота приёмной комиссии.

    :param user_id: Идентификатор пользователя.
    :param user_prompt: Запрос пользователя.
    :returns: Текстовый ответ от ассистента.
    """

    logger.info("Calling chatbot for query - '%s ...'", user_prompt[:100])
    async with AsyncSqliteSaver.from_conn_string(os.fspath(SQLITE_PATH)) as checkpointer:
        await checkpointer.setup()
        agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT.format(today_date=current_datetime()),
            context_schema=UserContext,
            middleware=[
                summarization_middleware,
                ToolCallLimitMiddleware(tool_name="web_search", run_limit=2),
                ToolCallLimitMiddleware(tool_name="browse_page", run_limit=2),
            ],
            tools=[remember, search_memory, search_knowledge, web_search, browse_page],
            checkpointer=checkpointer
        )
        result = await agent.ainvoke(
            {"messages": [("human", user_prompt)]},
            context=UserContext(user_id=user_id),
            config={"configurable": {"thread_id": user_id}}
        )
    return result["messages"][-1].content
