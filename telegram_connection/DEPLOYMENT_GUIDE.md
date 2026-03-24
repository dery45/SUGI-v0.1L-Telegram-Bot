# DEPLOYMENT GUIDE — Sugi v0.1L Telegram Bot

## Struktur File Baru

```
Local_RAG_Langchain/
├── sugi_core.py              ← Logic utama (Shared)
├── user_store.py             ← Management user (Shared)
├── .env                      ← Konfigurasi pusat (Root)
└── telegram_connection/      ← Folder khusus Telegram
    ├── telegram_bot.py       ← Entry point (Run this)
    └── requirements_telegram.txt
```

---

## Step 1 — Install Dependency

```bash
pip install -r telegram_connection/requirements_telegram.txt
```

---

## Step 2 — Buat Telegram Bot

1. Buka Telegram, cari **@BotFather**
2. Kirim `/newbot`
3. Ikuti instruksi, masukkan nama dan username bot
4. Salin **token** yang diberikan (format: `123456789:AAF...`)

---

## Step 3 — Set Token

Sugi menggunakan file `.env` di **root directory** untuk menyimpan token. 

Buka file `.env` di root proyek dan tambahkan/edit baris berikut:
```env
TELEGRAM_BOT_TOKEN="123456789:AAFxxxx"
```

Pastikan tidak ada spasi di sekitar tanda `=` dan token diapit tanda kutip.

---

## Step 4 — Jalankan Bot

```bash
# Pastikan terminal berada di root proyek (Local_RAG_Langchain)
python telegram_connection/telegram_bot.py
```

---

## Fitur Bot Telegram

| Command     | Fungsi                                          |
|-------------|------------------------------------------------|
| `/start`    | Mulai / restart, tampilkan tombol share kontak  |
| `/help`     | Panduan penggunaan                              |
| `/history`  | Lihat ringkasan percakapan sebelumnya           |
| `/clear`    | Reset sesi berjalan (memory lama tetap tersimpan)|
| `/contact`  | Update nomor HP                                 |
| `/about`    | Info tentang Sugi                               |

---

## Data yang Disimpan per User

**data/users.json** (lokal):
```json
{
  "123456789": {
    "user_id": "123456789",
    "platform": "telegram",
    "username": "petani_jawa",
    "full_name": "Budi Santoso",
    "phone_number": "+628123456789",
    "first_seen": "2025-01-01T10:00:00",
    "last_seen": "2025-01-15T14:30:00",
    "visit_count": 12,
    "session_ids": ["123456789_20250101_100000", ...]
  }
}
```

**ChromaDB (conversation_memory collection)**:
- Ringkasan percakapan per sesi
- Di-query saat user kembali untuk context continuity
- TTL: 14 hari (bisa diubah di sugi_core.py)

---

## Deploy ke Server (Opsional)

### Menggunakan systemd (Linux):

```ini
# /etc/systemd/system/sugi-bot.service
[Unit]
Description=Sugi Telegram Bot
After=network.target

[Service]
WorkingDirectory=/path/to/Local_RAG_Langchain
Environment=TELEGRAM_BOT_TOKEN=<token>
ExecStart=/path/to/venv/bin/python telegram_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable sugi-bot
sudo systemctl start sugi-bot
sudo systemctl status sugi-bot
```

### Menggunakan PM2 (Node.js process manager):

```bash
npm install -g pm2
pm2 start "python telegram_bot.py" --name sugi-bot
pm2 startup
pm2 save
```

---

## Catatan Penting

- **Ollama harus berjalan** sebelum bot dijalankan
- **ChromaDB server** harus aktif di `localhost:8000`
- Nomor HP hanya tersedia jika user **share contact** secara sukarela
- Bot hanya merespons pertanyaan **dalam scope pertanian**