"""
telegram_bot.py — Sugi v0.1L Telegram Bot Integration

Fitur:
  - Multi-user session via user_id Telegram
  - Riwayat percakapan persistent di ChromaDB
  - Nomor HP tersimpan jika user share contact
  - Context percakapan dilanjutkan di sesi berikutnya
  - Debug commands: !debug / !flags / !session / !memory / !stats
  - Command: /start /help /history /clear /contact /about /debug

Setup:
  pip install "python-telegram-bot>=21.0"
  Isi TELEGRAM_BOT_TOKEN di .env
"""

import sys
import os
import asyncio
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction

from sugi_core import SugiCore
from user_store import UserStore

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
MAX_MESSAGE_LENGTH = 4000   # Telegram limit 4096, sisakan buffer

# User yang diizinkan pakai debug commands (kosong = semua user bisa)
# Isi dengan Telegram user_id (string) jika ingin restrict
DEBUG_ALLOWED_USERS: set[str] = set(
    os.getenv("DEBUG_ALLOWED_USERS", "").split(",")
) - {""}


class SugiTelegramBot:

    def __init__(self):
        if not TELEGRAM_TOKEN:
            raise ValueError(
                "❌ TELEGRAM_BOT_TOKEN tidak ditemukan di .env!\n"
                "Isi nilai TELEGRAM_BOT_TOKEN di file .env"
            )
        self.sugi  = SugiCore()
        self.users = UserStore()
        print("🤖  Sugi Telegram Bot initialized.")

    # ── /start ────────────────────────────────────────────────────────────────
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user      = update.effective_user
        user_id   = str(user.id)
        user_info = self.users.get_or_create(
            user_id   = user_id,
            platform  = "telegram",
            username  = user.username or "",
            full_name = user.full_name or "",
        )

        is_returning = user_info.get("visit_count", 1) > 1
        if is_returning:
            last_visit = user_info.get("last_seen", "sebelumnya")
            greeting = (
                f"Halo lagi, {user.first_name}! 👋\n"
                f"Selamat datang kembali di Sugi v0.1L.\n"
                f"Kunjungan terakhirmu: {last_visit}\n\n"
                f"Ada yang bisa Sugi bantu hari ini? 🌾"
            )
        else:
            greeting = (
                f"Halo, {user.first_name}! 👋\n\n"
                f"Saya Sugi v0.1L, asisten pertanian digital Indonesia.\n\n"
                f"Saya siap membantu kamu seputar:\n"
                f"🌱 Pertanian & perkebunan\n"
                f"🌾 Ketahanan & harga pangan\n"
                f"🌤 Cuaca & iklim pertanian\n"
                f"🐛 Hama & penyakit tanaman\n"
                f"📈 Agribisnis & investasi\n\n"
                f"Ketik pertanyaanmu langsung, atau /help untuk panduan."
            )

        contact_btn = KeyboardButton("📱 Bagikan Nomor HP", request_contact=True)
        skip_btn    = KeyboardButton("⏭️ Lewati")
        markup = ReplyKeyboardMarkup(
            [[contact_btn], [skip_btn]],
            resize_keyboard=True, one_time_keyboard=True,
        )
        await update.message.reply_text(greeting, reply_markup=markup)

    # ── /help ─────────────────────────────────────────────────────────────────
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📖 Panduan Sugi Bot\n\n"
            "Command:\n"
            "/start   — Mulai atau restart percakapan\n"
            "/help    — Panduan ini\n"
            "/history — Riwayat percakapan sebelumnya\n"
            "/clear   — Reset sesi berjalan\n"
            "/contact — Perbarui nomor HP\n"
            "/about   — Info tentang Sugi\n"
            "/debug   — Panel debug (stats & logs)\n\n"
            "Cara bertanya:\n"
            "Ketik pertanyaan langsung, contoh:\n"
            "• Bagaimana cara menanam padi yang baik?\n"
            "• Harga cabai hari ini berapa?\n"
            "• Hama apa yang sering menyerang jagung?\n\n"
            "⚠️ Sugi hanya menjawab topik pertanian & pangan."
        )

    # ── /history ──────────────────────────────────────────────────────────────
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        summary = self.sugi.get_memory_summary(user_id)
        if not summary:
            await update.message.reply_text(
                "📭 Belum ada riwayat percakapan tersimpan.\n"
                "Mulailah bertanya, dan Sugi akan mengingat topik-topik penting!"
            )
            return
        await self._send_long(update, f"📚 Riwayat Percakapan Kamu:\n\n{summary}")

    # ── /clear ────────────────────────────────────────────────────────────────
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.sugi.clear_session(str(update.effective_user.id))
        await update.message.reply_text(
            "🗑️ Riwayat sesi ini telah dihapus.\n"
            "Percakapan baru dimulai! Apa yang ingin kamu tanyakan? 🌾"
        )

    # ── /about ────────────────────────────────────────────────────────────────
    async def cmd_about(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🌾 Sugi v0.1L\n\n"
            "Asisten AI pertanian Indonesia:\n"
            "• Model LLM lokal (Ollama)\n"
            "• RAG (Retrieval-Augmented Generation)\n"
            "• Hybrid search (BM25 + Vector)\n"
            "• Plant API (Perenual)\n"
            "• Weather integration\n"
            "• Eval loop (faithfulness + relevance)\n\n"
            "Dikembangkan oleh BU Team."
        )

    # ── /contact ──────────────────────────────────────────────────────────────
    async def cmd_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        contact_btn = KeyboardButton("📱 Bagikan Nomor HP", request_contact=True)
        markup = ReplyKeyboardMarkup(
            [[contact_btn]], resize_keyboard=True, one_time_keyboard=True
        )
        await update.message.reply_text(
            "Silakan bagikan nomor HPmu agar Sugi bisa memberikan layanan lebih personal.",
            reply_markup=markup,
        )

    # ── /debug ────────────────────────────────────────────────────────────────
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)

        # Cek akses jika DEBUG_ALLOWED_USERS di-set
        if DEBUG_ALLOWED_USERS and user_id not in DEBUG_ALLOWED_USERS:
            await update.message.reply_text("⛔ Akses debug tidak diizinkan.")
            return

        await update.message.reply_text(
            "🔧 Debug Panel\n\n"
            "Kirim salah satu command berikut:\n"
            "!debug  — Laporan 10 query terakhir\n"
            "!flags  — Query yang di-flag eval\n"
            "!session — Log sesi ini\n"
            "!memory — Memory tersimpan untukmu\n"
            "!stats  — Statistik SugiCore"
        )

    # ── Handle contact share ──────────────────────────────────────────────────
    async def handle_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        contact = update.message.contact
        user_id = str(update.effective_user.id)
        if contact:
            self.users.update_phone(user_id, contact.phone_number)
            await update.message.reply_text(
                f"✅ Nomor HP {contact.phone_number} berhasil disimpan.\n"
                f"Terima kasih! Sekarang kamu bisa langsung bertanya. 🌾"
            )
        else:
            await update.message.reply_text("⚠️ Tidak dapat membaca nomor HP.")

    # ── Handle tombol "Lewati" ────────────────────────────────────────────────
    async def handle_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Baik! Langsung saja ketik pertanyaanmu. 🌾",
            reply_markup=ReplyKeyboardRemove(),
        )

    # ── Handle pesan teks biasa ───────────────────────────────────────────────
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user     = update.effective_user
        user_id  = str(user.id)
        question = update.message.text.strip()

        self.users.get_or_create(
            user_id   = user_id,
            platform  = "telegram",
            username  = user.username or "",
            full_name = user.full_name or "",
        )
        self.users.update_last_seen(user_id)

        # ── [6] Debug commands via chat ───────────────────────────────────────
        if question.startswith("!"):
            # Cek akses jika DEBUG_ALLOWED_USERS di-set
            if DEBUG_ALLOWED_USERS and user_id not in DEBUG_ALLOWED_USERS:
                await update.message.reply_text("⛔ Akses debug tidak diizinkan.")
                return

            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            handled, output = await asyncio.to_thread(
                self.sugi.handle_debug_command, question, user_id
            )
            if handled:
                await self._send_long(update, output or "✅ Done.")
            else:
                await update.message.reply_text(
                    f"⚠️ Command tidak dikenal: {question}\n"
                    f"Tersedia: !debug / !flags / !session / !memory / !stats"
                )
            return

        # ── Pertanyaan biasa ──────────────────────────────────────────────────
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        response = await asyncio.to_thread(
            self.sugi.ask,
            user_id  = user_id,
            question = question,
            platform = "telegram",
        )

        await self._send_long(update, response)

    # ── Helper: kirim pesan panjang ───────────────────────────────────────────
    async def _send_long(self, update: Update, text: str):
        if not text:
            await update.message.reply_text("⚠️ Tidak ada respons dari Sugi.")
            return
        for i in range(0, len(text), MAX_MESSAGE_LENGTH):
            await update.message.reply_text(text[i:i+MAX_MESSAGE_LENGTH])

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        app = Application.builder().token(TELEGRAM_TOKEN).build()

        app.add_handler(CommandHandler("start",   self.cmd_start))
        app.add_handler(CommandHandler("help",    self.cmd_help))
        app.add_handler(CommandHandler("history", self.cmd_history))
        app.add_handler(CommandHandler("clear",   self.cmd_clear))
        app.add_handler(CommandHandler("about",   self.cmd_about))
        app.add_handler(CommandHandler("contact", self.cmd_contact))
        app.add_handler(CommandHandler("debug",   self.cmd_debug))

        app.add_handler(MessageHandler(filters.CONTACT,                 self.handle_contact))
        app.add_handler(MessageHandler(filters.Regex("^⏭️ Lewati$"),    self.handle_skip))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        print(f"🚀  Sugi Telegram Bot berjalan... (token: {TELEGRAM_TOKEN[:10]}...)")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    SugiTelegramBot().run()