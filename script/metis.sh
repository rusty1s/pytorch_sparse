#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "windows" ]; then
  choco install make
  cmake --help
fi

METIS=metis-5.1.0

wget -nv http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS}.tar.gz
tar -xvzf ${METIS}.tar.gz
cd ${METIS} || exit
sed -i.bak -e 's/IDXTYPEWIDTH 32/IDXTYPEWIDTH 64/g' include/metis.h
echo "CONFIG"
make config
echo "MAKE"
make
echo "MAKE INSTALL"
sudo make install
cd ..
