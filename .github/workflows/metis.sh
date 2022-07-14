#!/bin/bash

METIS=metis-5.1.0

wget -nv "https://web.archive.org/web/20211119110155/http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS}.tar.gz"
tar -xvzf "${METIS}.tar.gz"
rm -f "${METIS}.tar.gz"
cd "${METIS}" || exit
sed -i.bak -e 's/IDXTYPEWIDTH 32/IDXTYPEWIDTH 64/g' include/metis.h

make config
make
sudo make install

sudo cp /usr/local/include/metis.h /usr/include/
sudo cp /usr/local/lib/libmetis.a /usr/lib/

rm -f "${METIS}"
