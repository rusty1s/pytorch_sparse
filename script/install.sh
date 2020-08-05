#!/bin/bash

if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
  python setup.py develop
else
  echo "pip install .[test]"
  while sleep 30; do echo "=====[ Still running after $SECONDS seconds ]====="; done &
  python setup.py develop | tail -n 1000
  kill %1
fi
