"""Технические лимиты и настройки сэмплирования evidence, но не политика корзины."""

BLOB_MARKER_PATTERN = r"([A-ZА-ЯЁ][A-ZА-ЯЁ_]{1,15}):"

# Функции итоговой строки Excel: автосумма ставит SUBTOTAL на фильтрованных
# таблицах, MIN/MAX/MEDIAN — обычные сводки по колонке оценок (LAIM-0191).
AGGREGATE_FORMULA_PATTERN = (
    r"SUM|COUNT|AVERAGE|SUBTOTAL|MIN|MAX|MEDIAN"
    r"|СУММ|СЧЁТ|СЧЕТ|СРЗНАЧ|ПРОМЕЖУТОЧН|МИН|МАКС|МЕДИАН"
)

ERROR_PAYLOAD_CHAR_CAP = 2_000
MAX_TOKENS_CEILING = 65_536
EVIDENCE_CELL_CAP = 200
# Снимок книги для LLM: щедрые лимиты (токены не экономим), но независимые
# от числа строк корзины — растёт корзина, не снимок.
EVIDENCE_ANCHOR_ROWS = 10
EVIDENCE_SAMPLES_PER_COLUMN = 6
EVIDENCE_UNIQUES_MAX = 30
# Полный документ в промпте задачи: щедрый кап от окна модели, не от экономии.
DOCUMENT_CHAR_CAP = 120_000
