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


async def handle_photo_or_text(msg: types.Message):
    if msg.from_user.id not in ALLOWED_IDS:
        await msg.reply("⛔ Нет доступа")
        return

    if msg.text and msg.text.startswith("/"):
        return

    await bot.send_chat_action(msg.chat.id, "typing")

    try:
        content = []

        # Если есть фото
        if msg.photo:
            file_id = msg.photo[-1].file_id  # самое большое качество
            file = await bot.get_file(file_id)
            image_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path}"

            content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        # Если есть текст
        user_text = (msg.text or msg.caption or "").strip()
        if user_text:
            content.append({"type": "text", "text": user_text})
        elif msg.photo:
            content.append({"type": "text", "text": "Что на этом изображении? Распознай и ответь на русском."})

        if not content:
            return

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": content})

        response = await ai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )
        result = response.choices[0].message.content.strip()
        await msg.reply(result)
        logger.info("Запрос с фото: %r", user_text if user_text else "только фото")
    except Exception as e:
        logger.exception("OpenAI API error")
        await msg.reply(f"❌ Ошибка: {e}")


@dp.message(lambda msg: (msg.photo or msg.text) and not (msg.text and msg.text.startswith("/")))
async def handle_media(msg: types.Message):
    await handle_photo_or_text(msg)


async def main():
    logger.info("Бот запущен, ALLOWED_IDS=%s", ALLOWED_IDS)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановлен")
        sys.exit(0)
