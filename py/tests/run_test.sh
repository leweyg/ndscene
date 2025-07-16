
set -e # stop on error
cd "$(dirname "$0")" # cd to script directory
cd ../

# core tests
python3 -m tests.ndscene_tests

# pytorch testing (optional if this is removed):
python3 -m tests.ndscene_torch_tests
python3 -m tests.ndscene_torch_imageio


