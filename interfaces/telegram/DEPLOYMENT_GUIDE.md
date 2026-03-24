# DEPLOYMENT GUIDE — SUGI v0.1L Telegram Bot

## Implementation Structure

```
Local_RAG_Langchain/
├── core/
│   ├── sugi_core.py          ← Core Logic (Shared)
│   └── user_store.py         ← User Management (Shared)
├── config/
│   └── .env                  ← Central Configuration
└── interfaces/
    └── telegram/             ← Telegram Specific Folder
        ├── telegram_bot.py   ← Entry Point (Run this)
        └── requirements.txt
```

---

## Step 1 — Install Dependencies

```bash
pip install -r interfaces/telegram/requirements.txt
```

---

## Step 2 — Create your Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send the command `/newbot`
3. Follow the instructions to give your bot a name and username.
4. Copy the **token** provided (format: `123456789:AAF...`)

---

## Step 3 — Set the Token & Environment

SUGI uses a `.env` file in the **root directory** for all configurations. Use `.env.example` as a template.

### Key Environment Variables

- `TELEGRAM_BOT_TOKEN`: The API Token from BotFather.
- `DEBUG_ALLOWED_USERS`: A comma-separated list of Telegram user IDs (e.g., `123456,789012`). If empty, **all** users can use debug commands (`!debug`, `!session`, etc.). Use this to restrict access in production.
- `MEMORY_TTL_DAYS`: How many days to keep session summaries in ChromaDB (default: `14`).
- `SCOPE_CONFIG_PATH`: Path to the `.ini` file defining allowed agricultural topics.

---

## Step 4 — Launch the Bot

```bash
# Ensure your terminal is at the project root (Local_RAG_Langchain)
python interfaces/telegram/telegram_bot.py
```

---

## Bot Commands

| Command     | Description                                     |
|-------------|-------------------------------------------------|
| `/start`    | Start / Restart, prompts to share contact       |
| `/help`     | User guide                                      |
| `/history`  | View a summary of past conversations            |
| `/clear`    | Reset current session (old memory persists)     |
| `/contact`  | Update phone number                             |
| `/about`    | Information about SUGI                          |
| `/debug`    | (Admin) View current session performance        |

---

## Data Management

**user_store.py / data/users.json** (Local):
- Stores user profiles, phone numbers, and visit statistics.
- Shared between platforms (CLI and Bot).

**ChromaDB (conversation_memory collection)**:
- Stores session-based summaries.
- Queried when users return for context continuity.
- Managed via `MEMORY_TTL_DAYS` (default active).

---

## Server Deployment (Optional)

### Using systemd (Linux):

```ini
# /etc/systemd/system/sugi-bot.service
[Unit]
Description=SUGI Telegram Bot
After=network.target

[Service]
WorkingDirectory=/path/to/Local_RAG_Langchain
EnvironmentFile=/path/to/Local_RAG_Langchain/config/.env
ExecStart=/path/to/venv/bin/python interfaces/telegram/telegram_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Important Security Notes

1. **Debugging Safety**: Always set `DEBUG_ALLOWED_USERS` in production to prevent unauthorized access to query logs and session metadata.
2. **Context Window**: The system uses `num_ctx=4096` to prevent overflow errors. Ensure your local Ollama model supports this size.
3. **Embedding Limit**: Documents over 3,000 characters are automatically truncated during storage to fit within the `mxbai-embed-large` 512-token limit.
4. **Ollama Process**: Ollama must be running as a background service before starting the bot.