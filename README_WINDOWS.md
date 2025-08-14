
# BidWERX Prompt Lab — Windows Guide

Goal: Non-Technical Environment for Developing, Testing, and Refining Assistant API Prompts, Instructions, and Document-Context Responses.

This app runs in your browser and requires no coding. Use it to:
- Write/refine system instructions and prompts
- Upload PDF/DOCX/TXT documents and use them as context
- Run prompts and view AI responses side-by-side
- Export results to CSV/JSON for developer handoff

---

## Quick Start (Windows)

1. Install Python (once)
   - Download Python 3.10+ from https://www.python.org/downloads/windows/
   - During install, check "Add Python to PATH".

2. Open the project folder
   - Place all files in a folder like `C:\bidwerx-prompt-lab`
   - Double-click 'Start.bat'.

3. Provide your API key
   - When the app opens, use the left sidebar → paste your AI API Key.
   - The key is used only in your session, it is not saved.

4. Use the app
   - Enter System Instructions
   - Add one or more Prompts (with optional “Expected Result” notes)
   - Click Upload and select your document (PDF/DOCX/TXT)
   - Click Run Prompts
   - Review results, then click Export to download CSV/JSON

> Tip: For repeat use, just double-click Start.bat next time. It will re-activate your environment and launch.

---

## What’s Included

- `app.py` — the Streamlit app (opens in your browser)
- `requirements.txt` — dependencies
- `Start.bat` — one-click Windows launcher (Command Prompt)
- `README.md` — general instructions
- `README_WINDOWS.md` — this Windows-specific guide
---

## Security Notes

- **Do not** commit real keys to source control.
- Prefer using environment variables or a secrets manager for production.
- Keys entered in the sidebar are kept in-session only.

---

## Troubleshooting

- Streamlit not found — the launcher installs it for you automatically. Ensure internet connectivity.
- Browser doesn’t open — look in the console for a `Local URL`, e.g., `http://localhost:8501`, and open it manually.
- PDF parsing looks empty — some PDFs are image-based. Add OCR in a future version, or supply a DOCX/TXT version.
- Start.bat opens and instantly closes a terminal - open the Command Prompt, move into the folder with the files, then run "Start.bat" to see the error.
