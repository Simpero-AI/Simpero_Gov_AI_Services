"""Guards shared by the OOXML lanes (XLSX, DOCX).

Both formats are a zip of XML, so both carry the same untrusted-archive risks
and must be defended identically. One implementation, not one per lane: a
security control duplicated across formats is a control that drifts, and the
lane that quietly missed the fix is the one that gets fed the hostile file.
"""

from io import BytesIO
from zipfile import BadZipFile, ZipFile

from .errors import ParseError


def reject_decompression_bomb(archive_bytes: bytes, cap: int, *, kind: str) -> None:
    """Reject an OOXML file whose declared uncompressed size exceeds `cap`.

    Runs before the document is loaded: openpyxl/Docling both expand the archive
    into memory, so a decompression bomb (kilobytes on disk, gigabytes expanded)
    would OOM the worker. Sizes come from the zip's central directory, so nothing
    is decompressed here. `kind` ("XLSX"/"DOCX") only shapes the error message.
    """
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            uncompressed = sum(info.file_size for info in archive.infolist())
    except BadZipFile as exc:
        raise ParseError(
            f"corrupt_{kind.lower()}", f"Uploaded file is not a readable {kind}.", 400
        ) from exc
    if uncompressed > cap:
        raise ParseError(
            f"{kind.lower()}_too_large",
            f"{kind} expands to {uncompressed} bytes; maximum allowed is {cap}.",
            413,
        )
