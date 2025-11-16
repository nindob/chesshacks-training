# ChessHacks Bot

This repository contains:
- `my-chesshacks-bot/`: deployment code (serve.py, src, requirements)
- `training-suite/`: training pipelines and tools

To develop locally, follow the official ChessHacks development guide:
1. `python -m venv .venv` and `source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cd my-chesshacks-bot/devtools && npm install`
4. Configure `.env.local` files
5. Run `npm run dev` inside `devtools`

The bot loads weights from Hugging Face (set `HF_FILENAME` in `src/main.py`).

Deployment:
- `git add README.md`
- `git commit -m "Add README"`
- `git push`
