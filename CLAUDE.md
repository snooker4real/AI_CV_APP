# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered CV/resume reader application that uses PyPDF2 for PDF text extraction and supports both local Docker models and cloud AI providers for intelligent summarization. The application processes PDF documents by extracting text, chunking it for processing, and generating concise summaries using natural language processing.

**Default:** Uses llama3.2 running locally via Docker Model Runner (no API keys required)

## Architecture

**Single-file application:** The entire application is contained in `main.py` with a simple, linear architecture:

1. **PDFReader class** - Core component that handles:
   - PDF text extraction via PyPDF2
   - Text chunking for large documents (word-based splitting)
   - Multi-provider AI API integration (local Docker models + cloud APIs)
   - Multi-chunk summarization with recursive combining

2. **Processing flow:**
   - Extract text from PDF → Chunk if needed → Summarize each chunk → Combine summaries → Final refinement if needed

3. **AI Provider Integration:**

   **Local Models (via Docker Model Runner - NO API KEYS REQUIRED):**
   - **llama3.2** (DEFAULT) - 3.21B parameters, IQ2_XXS/Q4_K_M quantization
   - **deepseek-r1-distill-llama** - 8.03B parameters, IQ2_XXS/Q4_K_M quantization
   - **granite-docling** - 164.01M parameters, MOSTLY_F16 quantization, 8192 context
   - All use OpenAI-compatible API at http://localhost:8080/v1

   **Cloud Providers (require API keys):**
   - **OpenAI:** Uses gpt-3.5-turbo model
   - **DeepSeek:** Uses deepseek-chat model via OpenAI-compatible API (base_url: https://api.deepseek.com)
   - **Anthropic:** Uses claude-3-5-sonnet-20241022 model via native Anthropic API

   All configured with temperature 0.35 and max_tokens 300 for consistent outputs

## Environment Setup

**Python version:** 3.14.0

**Virtual environment:** `.venv/` directory contains project dependencies

**Dependencies:** PyPDF2, openai, anthropic, python-dotenv, pathlib

**Docker Model Runner Setup:**
```bash
# Enable in Docker Desktop settings
# Verify models are loaded:
docker model ls
```

**Environment variables (only for cloud providers):** `.env` file contains:
- `OPENAI_API_KEY` - Required when using OpenAI provider
- `DEEPSEEK_API_KEY` - Required when using DeepSeek provider
- `ANTHROPIC_API_KEY` - Required when using Anthropic provider

**Note:** Local Docker models (llama3, deepseek-r1, granite) do NOT require API keys.

## Running the Application

**Basic usage with llama3 (default, local, no API key needed):**
```bash
python main.py cv.pdf
```

**Using other local models:**
```bash
python main.py cv.pdf -p deepseek-r1
python main.py cv.pdf -p granite
```

**Using cloud providers:**
```bash
python main.py cv.pdf -p openai
python main.py cv.pdf -p deepseek
python main.py cv.pdf -p anthropic
```

**With all options:**
```bash
python main.py cv.pdf -p llama3 -l 300 -c 5000 -o output.txt
```

**CLI arguments:**
- `pdf_path` - Required: Path to PDF file
- `-p, --provider` - AI provider (default: llama3)
  - **Local:** `llama3` (default), `deepseek-r1`, `granite`
  - **Cloud:** `openai`, `deepseek`, `anthropic`
- `-l, --max-length` - Max summary length in words (default: 200)
- `-c, --chunk-size` - Chunk size for text processing (default: 4000)
- `-o, --output` - Output file path
- `--api-key` - Override API key from environment (cloud providers only)

## Development Notes

**Prerequisites for local models:**
- Docker Desktop must be running
- Model Runner must be enabled in Docker Desktop settings
- Models must be loaded (verify with `docker model ls`)

**Known issues in code:**
- Line ~117/133: Typo "nain ideas" should be "main ideas"
- Line 153: Typo "tet" should be "text"
- Line 156: Typo "founf" should be "found"
- Line 96/105: Missing spaces in string join operations
- Line 219: Uses `args.max_length` for `chunk_size` parameter (should be `args.chunk_size`)

**Text chunking behavior:** The `chunk_text` method splits by words but doesn't add spaces when joining, which creates concatenated text without proper spacing.

**API configuration:** Model uses temperature=0.35 and max_tokens=300 for consistent, focused summaries.
