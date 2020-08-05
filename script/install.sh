#!/bin/bash

if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
  python setup.py install
else
  echo "python setup.py install"
  while sleep 30; do echo "=====[ Still running after $SECONDS seconds ]====="; done &
  python setup.py install | tail -n 1000
  kill %1
fi
