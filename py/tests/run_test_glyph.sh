
set -e # stop on error
cd "$(dirname "$0")" # cd to script directory
cd ../

# core tests
echo "Glyph Tests..."
python3 -m tests.ndscene_glyph_test

echo "Glyph Test done."
