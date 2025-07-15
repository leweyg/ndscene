
set -e # stop on error
cd "$(dirname "$0")" # cd to script directory
./tests/run_test.sh
./examples/run.sh
