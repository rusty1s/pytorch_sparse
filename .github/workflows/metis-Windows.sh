#!/bin/bash

METIS=metis-5.1.0

curl -k -L "https://web.archive.org/web/20170712055800/http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS}.tar.gz" --output "${METIS}.tar.gz"
tar -xvzf "${METIS}.tar.gz"
rm -f "${METIS}.tar.gz"
cd "${METIS}" || exit
sed -i.bak -e 's/IDXTYPEWIDTH 32/IDXTYPEWIDTH 64/g' include/metis.h

# Fix GKlib on Windows: https://github.com/jlblancoc/suitesparse-metis-for-windows/issues/6
sed -i.bak -e '61,69d' GKlib/gk_arch.h

cd build || exit

cmake .. -A x64  # Ensure we are building with x64
cmake --build . --config "Release" --target ALL_BUILD
cp libmetis/Release/metis.lib /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Enterprise/VC/Tools/MSVC/14.29.30133/lib/x64
cp ../include/metis.h /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Enterprise/VC/Tools/MSVC/14.29.30133/include

rm -f "${METIS}"
