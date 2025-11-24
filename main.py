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

# Futuristic ASCII Art Banner
BANNER = """
[bold cyan]╔══════════════════════════════════════════════════════════════════════╗[/bold cyan]
║   [bright_magenta]██████╗ ██╗   ██║    ██████╗ ███████╗ █████╗ ██████╗ ███████╗██████╗[/bright_magenta]  ║
║   [bright_magenta]██╔══██╗██║   ██║    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗[/bright_magenta]  ║
║   [bright_cyan]██████╔╝██║   ██║    ██████╔╝█████╗  ███████║██║  ██║█████╗  ██████╔╝[/bright_cyan]  ║
║   [cyan]██╔══██╗╚██╗ ██╔╝    ██╔══██╗██╔══╝  ██╔══██║██║  ██║██╔══╝  ██╔══██╗[/cyan]  ║
║   [blue]██║  ██║ ╚████╔╝     ██║  ██║███████╗██║  ██║██████╔╝███████╗██║  ██║[/blue]  ║
║   [blue]╚═╝  ╚═╝  ╚═══╝      ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝[/blue]  ║
[bold cyan]╚══════════════════════════════════════════════════════════════════════╝[/bold cyan]

         [bold bright_magenta]◢◤[/bold bright_magenta] [bold white]CV READER v2.0[/bold white] [bold bright_magenta]◥◣[/bold bright_magenta]
            [dim bright_cyan]⟨ NEURAL EXTRACTION • DEEP ANALYSIS • SMART SUMMARIZATION ⟩[/dim bright_cyan]
"""


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

                console.print(f"\n[bold bright_cyan]▶ INITIATING NEURAL EXTRACTION PROTOCOL...[/bold bright_cyan]")
                console.print(f"[bright_magenta]⚡ SCANNING {total_pages} PAGE(S) ⚡[/bright_magenta]\n")

                text = ""
                with Progress(
                    SpinnerColumn(spinner_name="dots12"),
                    TextColumn("[bold bright_cyan]▸[/bold bright_cyan] [progress.description]"),
                    BarColumn(bar_width=40, style="bright_cyan", complete_style="bright_magenta", finished_style="bright_green"),
                    TaskProgressColumn(style="bright_white"),
                    console=console
                ) as progress:
                    task = progress.add_task("[bright_white]⟨ EXTRACTING DATA ⟩", total=total_pages)

                    for page_num, page in enumerate(reader.pages, 1):
                        text += page.extract_text() + "\n"
                        progress.update(task, advance=1, description=f"[bright_white]⟨ PAGE {page_num}/{total_pages} • PROCESSING ⟩")

                console.print(f"[bold bright_green]✓ EXTRACTION COMPLETE • DATA STREAM SECURED[/bold bright_green]\n")
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
        console.print(f"\n[bold bright_magenta]╔═══════════════════════════════════════════════════╗[/bold bright_magenta]")
        console.print(f"[bold bright_magenta]║[/bold bright_magenta]  [bold bright_white]⚡ QUANTUM A.I. ANALYSIS ENGINE ONLINE ⚡[/bold bright_white]  [bold bright_magenta]║[/bold bright_magenta]")
        console.print(f"[bold bright_magenta]╚═══════════════════════════════════════════════════╝[/bold bright_magenta]")
        console.print(f"[dim bright_cyan]⟨ MODEL: {self.model.upper()} • STATUS: ACTIVE ⟩[/dim bright_cyan]\n")

        text = self.extract_text(pdf_path)

        if not text.strip():
            return "⚠ NO DATA DETECTED IN QUANTUM FIELD"

        text_length = len(text)
        console.print(f"[bright_cyan]▸ DATA MATRIX SIZE: [bold bright_white]{text_length:,}[/bold bright_white] CHARS[/bright_cyan]\n")

        if text_length <= chunk_size:
            console.print("[bright_yellow]⟨ SINGLE QUANTUM PROCESSING MODE ENGAGED ⟩[/bright_yellow]")
            with console.status("[bold bright_green]⟨ A.I. NEURAL SYNTHESIS IN PROGRESS ⟩", spinner="bouncingBall"):
                return self.summarize_chunks(text, maxLength)

        chunks = self.chunk_text(text, chunk_size)
        num_chunks = len(chunks)
        console.print(f"[bright_yellow]▸ FRAGMENTING DATA INTO {num_chunks} QUANTUM CHUNK(S)[/bright_yellow]\n")

        summaries = []

        with Progress(
            SpinnerColumn(spinner_name="dots12"),
            TextColumn("[bold bright_cyan]▸[/bold bright_cyan] [progress.description]"),
            BarColumn(bar_width=40, style="bright_cyan", complete_style="bright_magenta", finished_style="bright_green"),
            TaskProgressColumn(style="bright_white"),
            console=console
        ) as progress:
            task = progress.add_task("[bright_white]⟨ NEURAL ANALYSIS ACTIVE ⟩", total=num_chunks)

            for i, chunk in enumerate(chunks, 1):
                chunk_max_length = maxLength // num_chunks
                progress.update(task, description=f"[bright_white]⟨ QUANTUM CHUNK {i}/{num_chunks} • SYNTHESIZING ⟩")

                summary = self.summarize_chunks(chunk, chunk_max_length)
                summaries.append(summary)
                progress.advance(task)

        combined_output = " ".join(summaries)

        if len(combined_output) > maxLength:
            console.print("\n[bright_yellow]⟨ INITIATING FINAL SYNTHESIS PROTOCOL ⟩[/bright_yellow]")
            with console.status("[bold bright_green]⟨ COMPRESSING NEURAL OUTPUT ⟩", spinner="bouncingBall"):
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

    # Print futuristic banner
    console.print(BANNER)

    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        console.print(Panel(
            f"[bold bright_red]⚠ CRITICAL ERROR: DOCUMENT NOT FOUND[/bold bright_red]\n\n"
            f"[bright_white]Path: {pdf_file}[/bright_white]\n"
            f"[dim]System cannot locate the specified PDF file in the quantum field.[/dim]",
            title="[bold bright_red]⚡ SYSTEM ALERT ⚡[/bold bright_red]",
            border_style="bright_red",
            box=box.HEAVY
        ))
        sys.exit(1)

    try:
        # Determine if using local or cloud model
        local_models = ["llama3", "deepseek-r1", "granite"]
        is_local = args.provider in local_models
        model_type = "◢ LOCAL QUANTUM ◣" if is_local else "☁ CLOUD NETWORK"

        # Create futuristic configuration table
        config_table = Table(
            title="[bold bright_magenta]⟨ SYSTEM CONFIGURATION ⟩[/bold bright_magenta]",
            box=box.HEAVY_HEAD,
            show_header=True,
            header_style="bold bright_cyan",
            border_style="bright_cyan",
            title_style="bold bright_magenta"
        )
        config_table.add_column("⚡ PARAMETER", style="bright_cyan", justify="right", no_wrap=True)
        config_table.add_column("◢ VALUE", style="bright_white", justify="left")

        config_table.add_row("▸ TARGET FILE", f"[bright_green]{pdf_file.name}[/bright_green]")
        config_table.add_row("▸ A.I. PROVIDER", f"[bright_magenta]{args.provider.upper()}[/bright_magenta]")
        config_table.add_row("▸ MODEL TYPE", f"[bright_yellow]{model_type}[/bright_yellow]")
        config_table.add_row("▸ OUTPUT LENGTH", f"[bright_white]{args.max_length}[/bright_white] [dim]words[/dim]")
        config_table.add_row("▸ QUANTUM CHUNK", f"[bright_white]{args.chunk_size}[/bright_white] [dim]chars[/dim]")

        console.print(config_table)

        read_CV = PDFReader(api_key=args.api_key, provider=args.provider)

        summary = read_CV.summarize_pdf(
            args.pdf_path,
            maxLength=args.max_length,
            chunk_size=args.chunk_size,
        )

        # Display summary in a futuristic panel
        console.print("\n")
        console.print("[bold bright_green]" + "═" * 70 + "[/bold bright_green]")
        summary_panel = Panel(
            f"[bright_white]{summary}[/bright_white]",
            title="[bold bright_green]⚡ ◢ NEURAL SYNTHESIS COMPLETE ◣ ⚡[/bold bright_green]",
            subtitle="[dim bright_cyan]⟨ ANALYSIS TERMINATED • DATA READY ⟩[/dim bright_cyan]",
            border_style="bright_green",
            box=box.DOUBLE_EDGE,
            padding=(1, 2)
        )
        console.print(summary_panel)
        console.print("[bold bright_green]" + "═" * 70 + "[/bold bright_green]")

        if args.output:
            output_file = Path(args.output)
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(summary)
            console.print(f"\n[bold bright_green]✓ DATA EXPORTED TO:[/bold bright_green] [bright_cyan]{output_file}[/bright_cyan]")
            console.print(f"[dim]⟨ File saved successfully to quantum storage ⟩[/dim]")

    except Exception as e:
        error_msg = str(e)
        console.print(f"\n[bold bright_red]⚠ ═══════════════════════════════════════════════════ ⚠[/bold bright_red]")
        console.print(Panel(
            f"[bold bright_red]SYSTEM MALFUNCTION DETECTED[/bold bright_red]\n\n"
            f"[bright_white]{error_msg}[/bright_white]",
            title="[bold bright_red]⚡ CRITICAL ERROR ⚡[/bold bright_red]",
            border_style="bright_red",
            box=box.HEAVY,
            padding=(1, 2)
        ))

        # Provide helpful hints for common errors
        local_models = ["llama3", "deepseek-r1", "granite"]
        if args.provider in local_models:
            if "docker model run failed" in error_msg.lower() or "no such file" in error_msg.lower():
                console.print(Panel(
                    "[bright_yellow]⚠ QUANTUM ENGINE OFFLINE ⚠[/bright_yellow]\n\n"
                    "[bright_white]Diagnostic Protocol:[/bright_white]\n"
                    "  [bright_cyan]▸[/bright_cyan] Verify Docker Desktop is running\n"
                    f"  [bright_cyan]▸[/bright_cyan] Check model availability: [dim]docker model ls[/dim]\n"
                    f"  [bright_cyan]▸[/bright_cyan] Test connection: [dim]docker model run {args.provider} \"Hello\"[/dim]\n\n"
                    "[dim]⟨ System requires active Docker Model Runner ⟩[/dim]",
                    title="[bold bright_yellow]⟨ TROUBLESHOOTING PROTOCOL ⟩[/bold bright_yellow]",
                    border_style="bright_yellow",
                    box=box.HEAVY_HEAD
                ))
            elif "timed out" in error_msg.lower():
                console.print(Panel(
                    "[bright_yellow]⚡ NEURAL TIMEOUT DETECTED ⚡[/bright_yellow]\n\n"
                    "[bright_white]The quantum processor exceeded maximum response time (>120s)[/bright_white]\n\n"
                    "[bright_cyan]▸ SOLUTION:[/bright_cyan] Reduce chunk size with [dim]-c[/dim] flag\n"
                    "[dim]⟨ Smaller chunks = faster processing ⟩[/dim]",
                    title="[bold bright_yellow]⟨ TIMEOUT ERROR ⟩[/bold bright_yellow]",
                    border_style="bright_yellow",
                    box=box.HEAVY_HEAD
                ))
        elif "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            console.print(Panel(
                "[bright_yellow]⚡ API QUOTA EXCEEDED ⚡[/bright_yellow]\n\n"
                "[bright_white]Cloud network resources depleted[/bright_white]\n\n"
                "[bright_cyan]▸ ACTION:[/bright_cyan] Verify billing and credits\n"
                "[dim]⟨ Contact your provider for quota increase ⟩[/dim]",
                title="[bold bright_yellow]⟨ QUOTA ERROR ⟩[/bold bright_yellow]",
                border_style="bright_yellow",
                box=box.HEAVY_HEAD
            ))
        elif "insufficient balance" in error_msg.lower() or "credit balance" in error_msg.lower():
            console.print(Panel(
                "[bright_yellow]⚡ INSUFFICIENT CREDITS ⚡[/bright_yellow]\n\n"
                "[bright_white]Account balance too low for operation[/bright_white]\n\n"
                "[bright_cyan]▸ ACTION:[/bright_cyan] Add credits to your account\n"
                "[dim]⟨ Recharge required to continue ⟩[/dim]",
                title="[bold bright_yellow]⟨ BALANCE ERROR ⟩[/bold bright_yellow]",
                border_style="bright_yellow",
                box=box.HEAVY_HEAD
            ))
        elif "api key" in error_msg.lower():
            console.print(Panel(
                "[bright_yellow]⚡ AUTHENTICATION FAILURE ⚡[/bright_yellow]\n\n"
                f"[bright_white]Missing or invalid API credentials[/bright_white]\n\n"
                f"[bright_cyan]▸ REQUIRED:[/bright_cyan] Set [bright_magenta]{args.provider.upper()}_API_KEY[/bright_magenta] in .env\n"
                "[dim]⟨ Secure quantum authentication required ⟩[/dim]",
                title="[bold bright_yellow]⟨ AUTH ERROR ⟩[/bold bright_yellow]",
                border_style="bright_yellow",
                box=box.HEAVY_HEAD
            ))

        console.print(f"[bold bright_red]⚠ ═══════════════════════════════════════════════════ ⚠[/bold bright_red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
