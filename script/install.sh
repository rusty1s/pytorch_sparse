#!/bin/bash

# if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
#   echo "LOL"
#   pip install .\[test\]
# else
echo "pip install .[test]"
while sleep 10; do echo "=====[ Still running after $SECONDS seconds ]====="; done &
pip install .\[test\] | tail -n 1000
kill %1
# fi
