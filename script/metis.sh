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
  # cp libmetis/Release/metis.lib ../..
  cp libmetis/Release/metis.lib /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/VC/Tools/MSVC/14.16.27023/lib/x64

  # cmake .. -DSHARED=1
  # cmake --build . --config "Release" --target ALL_BUILD
  # # cp libmetis/Release/metis.lib ../..
  # cp libmetis/Release/metis.dll /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/VC/Tools/MSVC/14.16.27023/bin

  # ls /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/VC/Tools/MSVC/14.16.27023/bin

  # cmake .. -DSHARED=1
  # cmake --build . --config "Release" --target ALL_BUILD
  # cp libmetis/Release/metis.dll /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/VC/Tools/MSVC/14.16.27023/bin/x64

  cp ../include/metis.h /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/VC/Tools/MSVC/14.16.27023/include

  cd ..
fi

cd ..
