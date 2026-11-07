"""Turn uploaded resume files (PDF / DOCX / TXT) into plain text."""
from __future__ import annotations

import io


class UnsupportedFile(ValueError):
    pass


def file_to_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:4] == b"%PDF":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        import docx

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    if name.endswith((".txt", ".md")) or not name:
        return data.decode("utf-8", errors="ignore")
    raise UnsupportedFile(f"Unsupported file type: {filename} (use .pdf, .docx or .txt)")
