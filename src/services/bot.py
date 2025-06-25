import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from session_manager import session_manager

from settings import settings
from prompts import prompt, base_prompt


dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}, write a request!")


@dp.message()
async def echo_handler(message: Message) -> None:
    try:

        session = await session_manager.get_session()

        context = await session.post(
            url=f"{settings.app_url}/api/v1/vectordb/search/",
            json={
                "query": f"{message.text}",
                "k": 5
            }
        )
        context = await context.json()

        message_answer = await message.answer("Запрос в обработке....")

        request = base_prompt.invoke(
            {
                "context": context[0]['content'],
                "metadata": context[0]["metadata"],
                "question": message.text,

            },
        )

        response = await session.post(
            url=settings.url_llm_model,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.api_key_llm_model}"
        },
            json={
                "prompt": request.text,
                "n": 1,
                "temperature": 0.8,
                "max_tokens": 2048
            }
        )

        response = await response.json()

        await message.answer(response["choices"][0]['text'])
        await message_answer.delete()
    except TypeError:
        await message.answer("Во время обработки запроса произошла ошибка...")


async def main() -> None:
    print("Starting bot...")
    bot = Bot(
        token=settings.api_token_telegram,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await dp.start_polling(bot)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bye!")
