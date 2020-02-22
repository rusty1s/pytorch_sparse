#!/bin/bash

METIS=metis-5.1.0

wget -nv http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS}.tar.gz
tar -xvzf ${METIS}.tar.gz
cd ${METIS} || exit
sed -i.bak -e 's/IDXTYPEWIDTH 32/IDXTYPEWIDTH 64/g' include/metis.h

if [ "${TRAVIS_OS_NAME}" != "windows" ]; then
  make config
  make
  sudo make install
else
  # Fix GKlib on Windows: https://github.com/jlblancoc/suitesparse-metis-for-windows/issues/6
  sed -i.bak -e '61,69d' GKlib/gk_arch.h

  cd build || exit

  cmake ..
  cmake --build . --config "Release" --target ALL_BUILD

  cp libmetis/Release/metis.lib /c/tools/miniconda3/envs/libs
  cp ../include/metis.h /c/tools/miniconda3/envs/include

  cd ..
fi

cd ..
