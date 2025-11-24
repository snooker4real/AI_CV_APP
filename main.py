# Build an AI powered CV Reader with Python -> PyPDF2 and Open AI

import  argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import PyPDF2
from openai import OpenAI # Get an OpenAI
from anthropic import Anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize Rich Console
console = Console()


# Main Class Analyze CV
class PDFReader:
    def __init__(self, api_key: Optional[str] = None, provider: str = "llama3"):
        """
        Initialize PDF Reader with specified AI provider.

        Args:
            api_key: API key for the provider (if not provided, reads from .env)
            provider: AI provider to use (local: "llama3", "deepseek-r1", "granite", cloud: "openai", "deepseek", "anthropic")
        """
        self.provider = provider.lower()
        self.is_local_model = False

        # Local Docker Model Runner models (using docker model run CLI)
        if self.provider in ["llama3", "llama3.2"]:
            self.is_local_model = True
            self.model = "llama3.2"
            self.client = None
        elif self.provider in ["deepseek-r1", "deepseek-r1-distill-llama"]:
            self.is_local_model = True
            self.model = "deepseek-r1-distill-llama"
            self.client = None
        elif self.provider == "granite":
            self.is_local_model = True
            self.model = "granite-docling"
            self.client = None
        # Cloud providers
        elif self.provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            self.model = "deepseek-chat"
        elif self.provider == "anthropic":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'llama3', 'deepseek-r1', 'granite', 'openai', 'deepseek', or 'anthropic'")

        # Only validate API keys for cloud providers (not needed for local models)
        if not self.is_local_model:
            if self.provider == "anthropic":
                if not api_key:
                    raise ValueError(f"{self.provider.upper()} API key is required")
            else:
                if not self.client.api_key:
                    raise ValueError(f"{self.provider.upper()} API key is required")

    # Extract Text
    def extract_text(self, pdf_path) -> List[str]:
        try:
            file = Path(pdf_path)

            with open(file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)

                console.print(f"[cyan]📄 Extracting text from {total_pages} page(s)...[/cyan]")

                text = ""
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("[green]Reading pages...", total=total_pages)

                    for page_num, page in enumerate(reader.pages, 1):
                        text += page.extract_text() + "\n"
                        progress.update(task, advance=1, description=f"[green]Reading page {page_num}/{total_pages}")

                return text.strip()
        except Exception as e:
            raise Exception(f"Error with PDF: {e}")

    # Chunk Text (Break up the text)
    def chunk_text(self, text:str, chunk_size:int= 4000) -> List[str]:
        words = text.split()

        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            if current_size + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # Summarize the chunks
    def summarize_chunks(self, chunks:str, maxLength:int = 200) -> str:
        try:
            if self.is_local_model:
                # Use docker model run CLI for local models with enhanced CV-specific context
                system_prompt = (
                    f"You are an expert HR recruiter analyzing a CV/resume. "
                    f"Summarize the following text in {maxLength} words or less.\n\n"
                    f"Focus on:\n"
                    f"- Professional experience and roles\n"
                    f"- Key skills and technical competencies\n"
                    f"- Educational background and certifications\n"
                    f"- Notable achievements and projects\n"
                    f"- Years of experience and career progression\n\n"
                    f"Be concise, professional, and highlight the most relevant information for recruitment purposes."
                )
                full_prompt = f"{system_prompt}\n\n---\n\nCV/Resume Text:\n{chunks}\n\n---\n\nSummary:"

                # Run docker model run command
                result = subprocess.run(
                    ["docker", "model", "run", self.model, full_prompt],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )

                if result.returncode != 0:
                    raise Exception(f"Docker model run failed: {result.stderr}")

                return result.stdout.strip()

            elif self.provider == "anthropic":
                # Anthropic API format with CV-specific context
                system_prompt = (
                    f"You are an expert HR recruiter analyzing a CV/resume. "
                    f"Summarize the following text in {maxLength} words or less. "
                    f"Focus on: professional experience and roles, key skills and technical competencies, "
                    f"educational background and certifications, notable achievements and projects, "
                    f"years of experience and career progression. "
                    f"Be concise, professional, and highlight the most relevant information for recruitment purposes."
                )
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    temperature=0.35,
                    system=system_prompt,
                    messages=[
                        {
                            "role":"user",
                            "content": f"CV/Resume Text:\n{chunks}\n\nProvide a professional summary:"
                        }
                    ]
                )
                return response.content[0].text.strip()
            else:
                # OpenAI API format (for openai and deepseek cloud) with CV-specific context
                system_prompt = (
                    f"You are an expert HR recruiter analyzing a CV/resume. "
                    f"Summarize the following text in {maxLength} words or less. "
                    f"Focus on: professional experience and roles, key skills and technical competencies, "
                    f"educational background and certifications, notable achievements and projects, "
                    f"years of experience and career progression. "
                    f"Be concise, professional, and highlight the most relevant information for recruitment purposes."
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role":"system",
                            "content": system_prompt,
                        },
                        {
                            "role":"user",
                            "content": f"CV/Resume Text:\n{chunks}\n\nProvide a professional summary:"
                        }
                    ],
                    max_tokens=300,
                    temperature=0.35,
                )
                return response.choices[0].message.content.strip()

        except subprocess.TimeoutExpired:
            raise Exception(f"Docker model run timed out after 120 seconds")
        except Exception as e:
            # Re-raise the exception so it can be handled at a higher level
            raise Exception(f"Error during summarization: {str(e)}")

    # Summarize the PDF
    def summarize_pdf(self, pdf_path:str, maxLength:int = 200, chunk_size:int = 400) -> str:
        console.print(f"\n[bold magenta]🤖 Starting AI Summarization[/bold magenta]")
        console.print(f"[dim]Using model: {self.model}[/dim]\n")

        text = self.extract_text(pdf_path)

        if not text.strip():
            return "No text found in PDF!"

        text_length = len(text)
        console.print(f"[cyan]📊 Extracted text length: {text_length:,} characters[/cyan]\n")

        if text_length <= chunk_size:
            console.print("[yellow]💡 Text fits in single chunk, processing...[/yellow]")
            with console.status("[bold green]Generating summary...", spinner="dots"):
                return self.summarize_chunks(text, maxLength)

        chunks = self.chunk_text(text, chunk_size)
        num_chunks = len(chunks)
        console.print(f"[yellow]✂️  Split into {num_chunks} chunk(s) for processing[/yellow]\n")

        summaries = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Summarizing chunks...", total=num_chunks)

            for i, chunk in enumerate(chunks, 1):
                chunk_max_length = maxLength // num_chunks
                progress.update(task, description=f"[cyan]Processing chunk {i}/{num_chunks}")

                summary = self.summarize_chunks(chunk, chunk_max_length)
                summaries.append(summary)
                progress.advance(task)

        combined_output = " ".join(summaries)

        if len(combined_output) > maxLength:
            console.print("\n[yellow]🔄 Refining combined summary...[/yellow]")
            with console.status("[bold green]Generating final summary...", spinner="dots"):
                return self.summarize_chunks(combined_output, maxLength)

        return combined_output

def main():
    parser = argparse.ArgumentParser(description="AI CV Reader with Rich TUI")

    parser.add_argument("pdf_path", help="Path to the PDF File")

    parser.add_argument("-l", "--max-length", type=int, default=200, help="Max summary length in words (default: 200)")

    parser.add_argument("-o", "--output", help="Output file path")

    parser.add_argument("-c", "--chunk-size", type=int, default=4000, help="Chunk size for text processing (default: 4000)")

    parser.add_argument("--api-key", help="API key for cloud providers (not needed for local models)")

    parser.add_argument("-p", "--provider",
                        choices=["llama3", "deepseek-r1", "granite", "openai", "deepseek", "anthropic"],
                        default="llama3",
                        help="AI provider to use. Local models (via Docker): llama3 (default), deepseek-r1, granite. Cloud models: openai, deepseek, anthropic")

    args = parser.parse_args()

    # Print welcome banner
    console.print(Panel.fit(
        "[bold cyan]AI-Powered CV/Resume Reader[/bold cyan]\n"
        "[dim]Extract, Analyze, and Summarize CVs with AI[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        console.print(f"[bold red]❌ Error:[/bold red] PDF file not found at: {pdf_file}")
        sys.exit(1)

    try:
        # Determine if using local or cloud model
        local_models = ["llama3", "deepseek-r1", "granite"]
        is_local = args.provider in local_models
        model_type = "Local (Docker)" if is_local else "Cloud API"

        # Create configuration table
        config_table = Table(title="Configuration", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        config_table.add_column("Setting", style="cyan", justify="right")
        config_table.add_column("Value", style="green")

        config_table.add_row("📄 PDF File", pdf_file.name)
        config_table.add_row("🤖 Provider", args.provider.upper())
        config_table.add_row("🏷️  Model Type", model_type)
        config_table.add_row("📏 Max Length", f"{args.max_length} words")
        config_table.add_row("✂️  Chunk Size", f"{args.chunk_size} chars")

        console.print(config_table)

        read_CV = PDFReader(api_key=args.api_key, provider=args.provider)

        summary = read_CV.summarize_pdf(
            args.pdf_path,
            maxLength=args.max_length,
            chunk_size=args.chunk_size,
        )

        # Display summary in a styled panel
        console.print("\n")
        summary_panel = Panel(
            summary,
            title="[bold green]✨ CV Summary[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(summary_panel)

        if args.output:
            output_file = Path(args.output)
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(summary)
            console.print(f"\n[bold green]✅ Summary saved to:[/bold green] [cyan]{output_file}[/cyan]")

    except Exception as e:
        error_msg = str(e)
        console.print(f"\n[bold red]❌ ERROR:[/bold red] {error_msg}\n")

        # Provide helpful hints for common errors
        local_models = ["llama3", "deepseek-r1", "granite"]
        if args.provider in local_models:
            if "docker model run failed" in error_msg.lower() or "no such file" in error_msg.lower():
                console.print(Panel(
                    "[yellow]Docker model run error. Please ensure:[/yellow]\n\n"
                    "  1. Docker Desktop is running\n"
                    f"  2. The model is available (run: [cyan]docker model ls[/cyan])\n"
                    f"  3. Test the model with: [cyan]docker model run {args.provider} \"Hello\"[/cyan]",
                    title="💡 Troubleshooting",
                    border_style="yellow",
                    box=box.ROUNDED
                ))
            elif "timed out" in error_msg.lower():
                console.print(Panel(
                    "[yellow]The model took too long to respond (>120 seconds).[/yellow]\n\n"
                    "Consider using a smaller chunk size with the [cyan]-c[/cyan] flag.",
                    title="⏱️  Timeout Issue",
                    border_style="yellow",
                    box=box.ROUNDED
                ))
        elif "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            console.print(Panel(
                "[yellow]This is an API quota issue.[/yellow]\n\n"
                "Please check your billing/credits for the selected provider.",
                title="💳 Quota Issue",
                border_style="yellow",
                box=box.ROUNDED
            ))
        elif "insufficient balance" in error_msg.lower() or "credit balance" in error_msg.lower():
            console.print(Panel(
                "[yellow]This is an API balance issue.[/yellow]\n\n"
                "Please add credits to your account.",
                title="💰 Balance Issue",
                border_style="yellow",
                box=box.ROUNDED
            ))
        elif "api key" in error_msg.lower():
            console.print(Panel(
                f"[yellow]Make sure your [cyan]{args.provider.upper()}_API_KEY[/cyan] is set in the .env file.[/yellow]",
                title="🔑 API Key Issue",
                border_style="yellow",
                box=box.ROUNDED
            ))

        sys.exit(1)


if __name__ == "__main__":
    main()
