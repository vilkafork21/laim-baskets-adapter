"""Публичный CLI: одна команда на один пакет."""

import argparse
import sys
import traceback
from pathlib import Path

from .errors import BasketError, LlmError
from .pipeline import run_package, write_failure


def _default_out(input_path: str) -> str:
    return str(Path("results") / Path(input_path).name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laim-basket")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="XLSX + 3 DOCX -> UMR + km.json")
    run.add_argument("input", help="папка одной корзины")
    run.add_argument("-o", "--out", help="папка результата")
    run.add_argument("--llm-preset", default="openrouter", choices=["openrouter", "contour"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = args.out or _default_out(args.input)
    try:
        result = run_package(args.input, out, args.llm_preset)
        if result.status == "computed":
            metric = result.km["main_metric"]
            print(f"run: {metric['name']} = {metric['value']} ({metric['threshold_verdict']})")
        else:
            print(f"run: not_evaluable: {result.km['reason']}")
        print(f"  Excel: {out}/{result.excel_name}")
        print(f"  KM:    {out}/km.json")
        return 0
    except LlmError as exc:
        write_failure(out, "run", exc)
        print(f"{exc.reason_code}: {exc}", file=sys.stderr)
        return 5
    except BasketError as exc:
        write_failure(out, "run", exc)
        print(f"{exc.reason_code}: {exc}", file=sys.stderr)
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_failure(out, "run", BasketError(str(exc), exception=type(exc).__name__))
        return 1


if __name__ == "__main__":
    sys.exit(main())
