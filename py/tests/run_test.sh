
set -e # stop on error
cd "$(dirname "$0")" # cd to script directory
cd ../

# core tests
echo "Core Tests..."
python3 -m tests.ndscene_tests
python3 -m tests.ndscene_glyph_test


# pytorch testing (optional if this is removed):
echo "PyTorch/ImageIO tests..."
python3 -m tests.ndscene_torch_tests
python3 -m tests.ndscene_torch_imageio


