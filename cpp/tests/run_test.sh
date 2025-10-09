
cd "$(dirname "$0")"

set -e 

./compile.sh

# cd to root of the repo (so that local paths match):
cd ../../

./cpp/tests/test_app

