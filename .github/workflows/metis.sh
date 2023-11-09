#!/bin/bash

METIS=metis-5.1.0

wget -nv "https://web.archive.org/web/20211119110155/http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS}.tar.gz"
tar -xvzf "${METIS}.tar.gz"
rm -f "${METIS}.tar.gz"
cd "${METIS}" || exit
sed -i.bak -e 's/IDXTYPEWIDTH 32/IDXTYPEWIDTH 64/g' include/metis.h
sed -i '1s/^/#if defined(__linux__) \&\& defined(__x86_64__)\n__asm__(".symver log,log@GLIBC_2.2.5");\n#endif\n/' GKlib/gk_proto.h
sed -i '1s/^/#if defined(__linux__) \&\& defined(__x86_64__)\n__asm__(".symver pow,pow@GLIBC_2.2.5");\n#endif\n/' libmetis/metislib.h

make config
make
sudo make install

sudo cp /usr/local/include/metis.h /usr/include/
sudo cp /usr/local/lib/libmetis.a /usr/lib/

rm -f "${METIS}"
