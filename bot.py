#!/usr/bin/env python3
"""QA Bot — Telegram → OpenAI GPT → ответ на любой вопрос (текст + голос)"""

import asyncio
import io
import logging
import os
import sys

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ALLOWED_IDS = [int(x.strip()) for x in os.environ.get("ALLOWED_IDS", "").split(",") if x.strip()]

# OpenAI
ai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Telegram
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

SYSTEM_PROMPT = """Ты — полезный ассистент. Отвечаешь на вопросы чётко, по делу, без лишней воды.

Правила:
- Отвечай на том же языке, на котором задан вопрос
- Если вопрос короткий — ответь коротко
- Если вопрос сложный — дай развёрнутый структурированный ответ
- Не используй Markdown разметку в ответах (ни жирный, ни курсив, ни код)
- Отвечай только на то, что спросили, без лишних отступлений"""


async def ask_gpt(question: str) -> str:
    response = await ai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


async def transcribe_voice(file_id: str) -> str:
    """Скачать голосовое из Telegram и распознать через Whisper API"""
    file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    buf.seek(0)
    buf.name = "voice.ogg"

    transcript = await ai_client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        language="ru",
    )
    return transcript.text.strip()


@dp.message(Command("start"))
async def cmd_start(msg: types.Message):
    if msg.from_user.id not in ALLOWED_IDS:
        await msg.reply("⛔ Нет доступа")
        return
    await msg.reply(
        "👋 Привет! Я Q&A ассистент на GPT-4o.\n"
        "Пиши текст или отправляй голосовые сообщения — я отвечу."
    )


@dp.message(lambda msg: msg.voice is not None)
async def handle_voice(msg: types.Message):
    if msg.from_user.id not in ALLOWED_IDS:
        await msg.reply("⛔ Нет доступа")
        return

    await bot.send_chat_action(msg.chat.id, "typing")

    try:
        # Распознаём речь
        text = await transcribe_voice(msg.voice.file_id)
        logger.info("Голос распознан: %r", text[:80])

        # Отвечаем
        result = await ask_gpt(text)
        await msg.reply(result)
    except Exception as e:
        logger.exception("Voice processing error")
        await msg.reply(f"❌ Ошибка: {e}")


@dp.message()
async def handle_text(msg: types.Message):
    if msg.from_user.id not in ALLOWED_IDS:
        await msg.reply("⛔ Нет доступа")
        return

    if not msg.text or msg.text.startswith("/"):
        return

    await bot.send_chat_action(msg.chat.id, "typing")

    try:
        result = await ask_gpt(msg.text.strip())
        await msg.reply(result)
        logger.info("Вопрос: %r", msg.text[:80])
    except Exception as e:
        logger.exception("OpenAI API error")
        await msg.reply(f"❌ Ошибка: {e}")


async def main():
    logger.info("Бот запущен, ALLOWED_IDS=%s", ALLOWED_IDS)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановлен")
        sys.exit(0)
