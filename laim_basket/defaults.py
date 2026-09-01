"""Технические лимиты и настройки сэмплирования evidence, но не политика корзины."""

BLOB_MARKER_PATTERN = r"([A-ZА-ЯЁ][A-ZА-ЯЁ_]{1,15}):"
BLOB_MARKER_MIN_SHARE = 0.9
BLOB_MIN_ROWS = 3

VOLATILE_FORMULA_PATTERN = r"RAND|СЛЧИС"
AGGREGATE_FORMULA_PATTERN = r"SUM|COUNT|AVERAGE|СУММ|СЧЁТ|СЧЕТ|СРЗНАЧ"

EVIDENCE_PROMPT_CHAR_CAP = 300_000
ERROR_PAYLOAD_CHAR_CAP = 2_000
MAX_TOKENS_CEILING = 65_536
EVIDENCE_SAMPLE_ROWS = 3
EVIDENCE_CELL_CAP = 200
EVIDENCE_UNIQUES_MAX = 30
EVIDENCE_UNIQUE_CAP = 80
EVIDENCE_HEADER_PREVIEW_ROWS = 5
# Вводная часть документов в layout-промпте: её хватает, чтобы отличить книгу
# одного агента от книги другого, и она не вытесняет workbook evidence.
DOCUMENT_INTRO_SPANS = 40
DOCUMENT_INTRO_CHAR_CAP = 6_000
