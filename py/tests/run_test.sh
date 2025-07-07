
set -e # stop on error
cd "$(dirname "$0")" # cd to script directory
cd ../
python3 -m tests.ndscene_tests

