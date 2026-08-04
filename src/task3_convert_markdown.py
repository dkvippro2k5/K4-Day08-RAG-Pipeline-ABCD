"""
Task 3 — Convert toàn bộ file trong ``data/landing/`` thành Markdown.

Sử dụng MarkItDown của Microsoft cho các định dạng tài liệu (PDF, DOCX,
HTML, TXT...). Các bài viết crawl được lưu dưới dạng JSON và có sẵn trường
``content_markdown``; với loại này, script giữ nguyên nội dung Markdown và
đưa metadata của bài viết vào phần đầu file.

Cài đặt:
    pip install "markitdown[pdf]"

Output luôn giữ nguyên đường dẫn tương đối bên dưới ``data/landing/``:
``data/landing/legal/policy.pdf`` → ``data/standardized/legal/policy.md``.
"""

import json
from pathlib import Path
from typing import Any

try:
    from markitdown import MarkItDown
except ImportError:  # pragma: no cover - chỉ xảy ra khi thiếu dependency môi trường
    MarkItDown = None  # type: ignore[assignment,misc]


PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"

# Đây là các định dạng đầu vào được dùng trong lab. Những file marker như
# ``.gitkeep`` và các file không thuộc danh sách này sẽ được bỏ qua.
SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".doc", ".docx", ".html", ".htm", ".json", ".txt", ".md"}
)


def _iter_source_files(directory: Path) -> list[Path]:
    """Trả về các file đầu vào được hỗ trợ theo thứ tự ổn định."""
    if not directory.exists():
        return []

    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.relative_to(directory).as_posix().lower(),
    )


def _new_markitdown() -> Any:
    """Khởi tạo MarkItDown và báo lỗi cài đặt đủ rõ ràng."""
    if MarkItDown is None:
        raise RuntimeError(
            'Thiếu dependency MarkItDown. Hãy chạy: pip install "markitdown[pdf]"'
        )
    return MarkItDown()


def _output_path(source_path: Path) -> Path:
    """Tính đường dẫn output, giữ nguyên cấu trúc tương đối của input."""
    relative_path = source_path.resolve().relative_to(LANDING_DIR.resolve())
    return (OUTPUT_DIR / relative_path).with_suffix(".md")


def _as_text(value: Any, default: str = "N/A") -> str:
    """Chuyển metadata JSON về text an toàn để đưa vào Markdown."""
    if value is None or value == "":
        return default
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _json_content(data: Any) -> tuple[str, str, str, str]:
    """Lấy title/source/date/content từ JSON crawl hoặc JSON thông thường."""
    if isinstance(data, dict):
        title = _as_text(data.get("title"), "Untitled")
        source = _as_text(data.get("url"), "N/A")
        crawled = _as_text(
            data.get("date_crawled", data.get("date")),
            "N/A",
        )

        content: Any = None
        for key in ("content_markdown", "content", "markdown", "text", "body"):
            if key in data and data[key] is not None:
                content = data[key]
                break

        if content is None:
            # Vẫn tạo ra Markdown hợp lệ cho JSON không theo schema crawl.
            content = "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2)

        return title, source, crawled, str(content).strip()

    # JSON root là list/scalar: giữ nguyên dữ liệu trong code fence.
    return (
        "Untitled",
        "N/A",
        "N/A",
        "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```",
    )


def _convert_json(filepath: Path) -> str:
    """Convert JSON crawl thành Markdown có metadata nguồn."""
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được JSON {filepath}: {exc}") from exc

    title, source, crawled, content = _json_content(data)
    header = (
        f"# {title}\n\n"
        f"**Source:** {source}\n"
        f"**Crawled:** {crawled}\n\n"
        "---\n\n"
    )
    return header + content + "\n"


def _convert_with_markitdown(filepath: Path, converter: Any) -> str:
    """Convert một file tài liệu bằng MarkItDown và lấy text_content."""
    try:
        result = converter.convert(str(filepath))
    except Exception as exc:
        raise RuntimeError(f"Không thể convert {filepath}: {exc}") from exc

    text_content = getattr(result, "text_content", result)
    if text_content is None:
        return ""
    return str(text_content).rstrip() + "\n"


def convert_file(filepath: Path, converter: Any | None = None) -> Path:
    """Convert một file đầu vào và trả về đường dẫn Markdown đã ghi."""
    filepath = Path(filepath)
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Định dạng chưa được hỗ trợ: {filepath.suffix or filepath.name}")

    if filepath.suffix.lower() == ".json":
        content = _convert_json(filepath)
    else:
        content = _convert_with_markitdown(
            filepath,
            converter if converter is not None else _new_markitdown(),
        )

    output_path = _output_path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _convert_directory(directory: Path) -> list[Path]:
    """Convert các file được hỗ trợ trong một thư mục và các thư mục con."""
    files = _iter_source_files(directory)
    if not files:
        return []

    converter = None
    converted: list[Path] = []
    for filepath in files:
        relative = filepath.resolve().relative_to(LANDING_DIR.resolve())
        print(f"Converting: {relative.as_posix()}")

        # JSON crawl không cần khởi tạo MarkItDown; các file còn lại dùng
        # chung một instance để tránh nạp converter nhiều lần.
        if filepath.suffix.lower() != ".json" and converter is None:
            converter = _new_markitdown()

        output_path = convert_file(filepath, converter)
        converted.append(output_path)
        print(f"  ✓ Saved: {output_path}")

    return converted


def convert_legal_docs() -> list[Path]:
    """Convert tài liệu PDF/DOC/DOCX (và định dạng được hỗ trợ khác) trong legal/."""
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    return _convert_directory(LANDING_DIR / "legal")


def convert_news_articles() -> list[Path]:
    """Convert JSON/HTML/Markdown bài crawl trong news/."""
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    return _convert_directory(LANDING_DIR / "news")


def convert_all() -> list[Path]:
    """Quét và convert toàn bộ input được hỗ trợ bên dưới data/landing/."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted = _convert_directory(LANDING_DIR)

    print(f"\n✓ Done! Đã convert {len(converted)} file.")
    print("  Output tại:", OUTPUT_DIR)
    return converted


if __name__ == "__main__":
    convert_all()
