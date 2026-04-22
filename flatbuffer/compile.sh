
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_ROOT="$REPO_ROOT/py"
TS_ROOT="$REPO_ROOT/ts"
JS_ROOT="$REPO_ROOT/js"
SCHEMA_PATH="$REPO_ROOT/flatbuffer/ndscene.fbs"
TS_RUNTIME_PACKAGE="$TS_ROOT/node_modules/flatbuffers"
JS_RUNTIME_PACKAGE="$JS_ROOT/node_modules/flatbuffers"

# Python from FlatBuffer
cd "$PY_ROOT"
uv run ndscenepy/flatbuffers/compiler.py

# TypeScript from FlatBuffer
cd "$REPO_ROOT"
flatc --ts -o "$TS_ROOT" "$SCHEMA_PATH"
tsc -p "$REPO_ROOT/tsconfig.flatbuffers.json"

# JavaScript from TypeScript
mkdir -p "$JS_RUNTIME_PACKAGE"
cp "$TS_RUNTIME_PACKAGE/package.json" "$JS_RUNTIME_PACKAGE/package.json"
