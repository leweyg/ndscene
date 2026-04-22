
set -e
cd "$(dirname "$(realpath "$0")")"
cd ..
cd py
uv run ndscenepy/flatbuffers/compiler.py
# now check py/ndscenepy/flatbuffers/generated/ndscene/NDBuffer.py
