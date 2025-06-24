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
        print(f"query - {message.text}")
        session = await session_manager.get_session()

        context = await session.post(
            url=f"{settings.app_url}/api/v1/vectordb/search/",
            json={
                "query": f"{message.text}",
                "k": 2
            }
        )
        context = await context.json()

        print(f"context response - {context}")

        request = base_prompt.invoke(
            {
                "context": context,
                "metadata": "metadata",
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
                "prompt": request.text,#f"Prompt: {prompt}\nВопрос: {message.text}\nКонтекст: {context}",
                "n": 1,
                "temperature": 0.8,
                "max_tokens": 2048
            }
        )
        print(f"отправлен")
        response = await response.json()
        print(f"response llm - {response}")
        await message.answer(response["choices"][0]['text'])

    except TypeError:
        await message.answer("Nice try!")


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
