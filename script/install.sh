#!/bin/bash

if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
  python setup.py install
else
  echo "python setup.py install"
  while sleep 9m; do echo "=====[ $SECONDS seconds still running ]====="; done &
  python setup.py install &> /dev/null
fi
