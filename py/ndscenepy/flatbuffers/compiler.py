import argparse
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _schema_path() -> Path:
    return _repo_root() / "flatbuffer" / "ndscene.fbs"


def _python_output_root() -> Path:
    return _repo_root() / "py"


def _generated_package_root() -> Path:
    return _python_output_root() / "ndscenepy" / "flatbuffers" / "generated"


def _ensure_package_files():
    package_paths = [
        _python_output_root() / "ndscenepy" / "flatbuffers",
        _generated_package_root(),
        _generated_package_root() / "ndscene",
    ]
    for path in package_paths:
        path.mkdir(parents=True, exist_ok=True)
        init_path = path / "__init__.py"
        if not init_path.exists():
            init_path.write_text('"""FlatBuffers package."""\n', encoding="utf-8")


def build_command(flatc: str = "flatc") -> list[str]:
    return [
        flatc,
        "--python",
        "-o",
        str(_python_output_root()),
        str(_schema_path()),
    ]


def compile_schema(flatc: str = "flatc", check: bool = True) -> int:
    _ensure_package_files()
    command = build_command(flatc=flatc)
    completed = subprocess.run(command, check=False)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compile flatbuffer/ndscene.fbs into the ndscenepy package.")
    parser.add_argument("--flatc", default="flatc", help="Path to the FlatBuffers compiler executable.")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the flatc command instead of running it.",
    )
    args = parser.parse_args(argv)

    _ensure_package_files()
    command = build_command(flatc=args.flatc)
    if args.print_command:
        print(" ".join(command))
        return 0
    return compile_schema(flatc=args.flatc, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
