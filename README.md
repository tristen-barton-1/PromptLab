
# BidWERX Prompt Lab (MVP)

A no-code Streamlit app for Product Owners to write/refine system instructions and prompts, upload a document, and test AI responses with the document content as context.

## Features (MVP)
- System instructions editor
- Multiple prompts input (with optional expected result notes)
- Upload PDF/DOCX/TXT; automatic parsing
- Run prompts with document context (first-chunk for speed)
- View responses aligned with prompts/instructions
- Export results to CSV/JSON
- Works with OpenAI-compatible chat completions (uses `OPENAI_API_KEY`)

## Quick Start
1. Create a virtual environment (optional)  
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install deps
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API key  
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   > You can also paste the key in the app sidebar for local testing.
4. Run the app  
   ```bash
   streamlit run app.py
   ```
5. Use it  
   - Enter system instructions and prompts  
   - Upload a document  
   - Click Run Prompts  
   - Review results and Download CSV/JSON