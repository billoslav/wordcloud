#!/usr/bin/env python3
"""
Lightweight CLI to generate a wordcloud image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wordcloud import Wordcloud
from wordcloud.utils import STRATEGIES
from wordcloud.utils.visualization import COLOR_THEMES
from wordcloud.utils.export import export_batch, normalize_export_formats, SUPPORTED_EXPORT_FORMATS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a wordcloud image from text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from text string
  wordcloud --text "Hello world hello" --output cloud.png

  # Generate from file
  wordcloud --file input.txt --output cloud.png --width 800 --height 600

  # Use color theme and placement strategy
  wordcloud --file input.txt --color-theme viridis --strategy rectangular --output cloud.png

  # Export multiple formats
  wordcloud --file input.txt --formats png,svg,html --output Results/cloud
        """.strip()
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-t", "--text",
        help="Text string to generate wordcloud from"
    )
    input_group.add_argument(
        "-f", "--file",
        type=Path,
        help="Path to text file to generate wordcloud from"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="wordcloud.png",
        help="Output file path (default: wordcloud.png)"
    )
    parser.add_argument(
        "--formats",
        help=f"Comma-separated output formats (e.g. png,svg,html). Supported: {', '.join(sorted(SUPPORTED_EXPORT_FORMATS))}"
    )
    
    # Image dimensions
    parser.add_argument(
        "--width",
        type=int,
        default=600,
        metavar="PIXELS",
        help="Image width in pixels (default: 600)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=338,
        metavar="PIXELS",
        help="Image height in pixels (default: 338)"
    )
    
    # Word filtering
    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        metavar="N",
        help="Maximum number of words to include (default: 200)"
    )
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=3,
        metavar="N",
        help="Minimum word length to include (default: 3)"
    )
    parser.add_argument(
        "--stopwords",
        help="Comma-separated list of stopwords to exclude"
    )
    
    # Font and styling
    parser.add_argument(
        "--font",
        dest="font_path",
        default="fonts/Arial Unicode.ttf",
        help="Path to font file (default: fonts/Arial Unicode.ttf)"
    )
    parser.add_argument(
        "--background",
        default="white",
        help="Background color (default: white)"
    )
    parser.add_argument(
        "--color-theme",
        choices=list(COLOR_THEMES.keys()),
        help=f"Color theme to use. Available: {', '.join(COLOR_THEMES.keys())}"
    )
    parser.add_argument(
        "--black-white",
        action="store_true",
        help="Use only black text (overrides color theme)"
    )

    # Rotation / orientation
    parser.add_argument(
        "--prefer-horizontal",
        type=float,
        default=1.0,
        metavar="P",
        help="Probability of keeping words horizontal (0.0..1.0). Default: 1.0",
    )
    parser.add_argument(
        "--rotate-angles",
        default="90,-90",
        metavar="LIST",
        help='Comma-separated rotation angles in degrees (e.g., "0,45,90,-90"). Default: "90,-90".',
    )
    
    # Text processing options
    parser.add_argument(
        "--language",
        help="Language code (e.g., 'en', 'fr', 'zh'). If not specified, will auto-detect."
    )
    parser.add_argument(
        "--enable-stemming",
        action="store_true",
        help="Enable stemming (requires NLTK)"
    )
    parser.add_argument(
        "--enable-lemmatization",
        action="store_true",
        help="Enable lemmatization (requires NLTK or spaCy)"
    )
    parser.add_argument(
        "--n-gram-range",
        metavar="MIN,MAX",
        help="N-gram range, e.g., '1,2' for unigrams and bigrams"
    )
    
    # Typography options
    parser.add_argument(
        "--font-distribution",
        choices=['linear', 'logarithmic', 'power'],
        default='linear',
        help="Font size distribution method (default: linear)"
    )
    parser.add_argument(
        "--font-distribution-exponent",
        type=float,
        default=0.5,
        help="Exponent for power-law font distribution (default: 0.5)"
    )
    
    # Placement strategy
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="random",
        help=f"Word placement strategy (default: random). Available: {', '.join(STRATEGIES)}"
    )
    
    # Advanced options
    parser.add_argument(
        "--min-font-size",
        type=int,
        default=14,
        help="Minimum font size in points (default: 14)"
    )
    parser.add_argument(
        "--max-font-size",
        type=int,
        help="Maximum font size in points (default: auto)"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=2,
        help="Margin between words in pixels (default: 2)"
    )
    
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate command-line arguments."""
    if args.width <= 0 or args.height <= 0:
        parser.error("Width and height must be positive integers")
    
    if args.max_words <= 0:
        parser.error("--max-words must be positive")
    
    if args.min_word_length <= 0:
        parser.error("--min-word-length must be positive")
    
    if args.min_font_size <= 0:
        parser.error("--min-font-size must be positive")
    
    if args.max_font_size and args.max_font_size < args.min_font_size:
        parser.error("--max-font-size must be >= --min-font-size")
    
    if args.margin < 0:
        parser.error("--margin must be non-negative")

    if not (0.0 <= float(args.prefer_horizontal) <= 1.0):
        parser.error("--prefer-horizontal must be between 0.0 and 1.0")

    # Validate rotate angles early (now supports any angle)
    try:
        angles = parse_rotate_angles(args.rotate_angles)
    except Exception as e:
        parser.error(f"--rotate-angles invalid: {e}")
    for a in angles:
        if not isinstance(a, int) or a < -360 or a > 360:
            parser.error(f"--rotate-angles: angles must be integers in [-360, 360] range, got {a}")
    
    # Validate n-gram range
    if args.n_gram_range:
        try:
            parts = args.n_gram_range.split(',')
            if len(parts) != 2:
                raise ValueError("Expected two values")
            min_n, max_n = int(parts[0].strip()), int(parts[1].strip())
            if min_n < 1 or max_n < min_n:
                raise ValueError("Invalid range")
        except Exception as e:
            parser.error(f"--n-gram-range invalid: {e}. Expected format: '1,2'")
    
    # Check font file exists
    font_path = Path(args.font_path)
    if not font_path.exists():
        parser.error(f"Font file not found: {args.font_path}")

    if args.formats:
        try:
            parse_formats(args.formats, args.output)
        except ValueError as e:
            parser.error(str(e))


def parse_rotate_angles(value: str) -> tuple[int, ...]:
    """Parse comma-separated rotation angles into a tuple of ints."""
    raw = [v.strip() for v in (value or "").split(",") if v.strip()]
    if not raw:
        raise ValueError("rotate angles list is empty")
    angles: list[int] = []
    for item in raw:
        angles.append(int(item))
    return tuple(angles)


def load_text(args: argparse.Namespace) -> str:
    """Load text from file or use provided text string."""
    if args.text:
        return args.text
    if args.file:
        if not args.file.exists():
            raise FileNotFoundError(f"File not found: {args.file}")
        try:
            return args.file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                return args.file.read_text(encoding="latin-1")
            except Exception as e:
                raise ValueError(f"Could not read file {args.file}: {e}")
    raise ValueError("Provide either --text or --file")


def parse_stopwords(stopwords_str: str | None) -> list[str]:
    """Parse comma-separated stopwords string into list."""
    if not stopwords_str:
        return []
    return [word.strip().lower() for word in stopwords_str.split(",") if word.strip()]


def parse_formats(formats_str: str | None, output_path: Path) -> list[str]:
    if formats_str:
        formats = formats_str.split(",")
        return normalize_export_formats(formats)
    if output_path.suffix:
        return normalize_export_formats([output_path.suffix.lstrip(".")])
    return ["png"]


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        validate_args(args, parser)
    except argparse.ArgumentError as e:
        parser.error(str(e))
    
    try:
        text = load_text(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if not text.strip():
        print("Error: Input text is empty", file=sys.stderr)
        return 1
    
    stopwords = parse_stopwords(args.stopwords)
    try:
        rotation_angles = parse_rotate_angles(args.rotate_angles)
    except Exception as e:
        print(f"Error: invalid --rotate-angles: {e}", file=sys.stderr)
        return 1
    
    # Parse n-gram range
    n_gram_range = None
    if args.n_gram_range:
        parts = args.n_gram_range.split(',')
        n_gram_range = (int(parts[0].strip()), int(parts[1].strip()))
    
    # Parse font distribution params
    font_distribution_params = {}
    if args.font_distribution == 'power':
        font_distribution_params['exponent'] = args.font_distribution_exponent
    
    try:
        wc = Wordcloud(
            width=args.width,
            height=args.height,
            max_words=args.max_words,
            min_word_length=args.min_word_length,
            font_path=str(args.font_path),
            place_strategy=args.strategy,
            background_color=args.background,
            stopwords=stopwords,
            min_font_size=args.min_font_size,
            max_font_size=args.max_font_size,
            margin=args.margin,
            black_white=args.black_white,
            prefer_horizontal=float(args.prefer_horizontal),
            rotation_angles=rotation_angles,
            language=args.language,
            enable_stemming=args.enable_stemming,
            enable_lemmatization=args.enable_lemmatization,
            n_gram_range=n_gram_range,
            font_distribution=args.font_distribution,
            font_distribution_params=font_distribution_params if font_distribution_params else None,
        )
        
        wc.generate(text, color_theme=(args.color_theme if not args.black_white else None))

        formats = parse_formats(args.formats, args.output)
        results = export_batch(wc, args.output, formats)
        for fmt, path in results.items():
            print(f"{fmt.upper()} saved to: {path.resolve()}")
        return 0
        
    except Exception as e:
        print(f"Error generating wordcloud: {e}", file=sys.stderr)
        if "--debug" in (argv or []):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

