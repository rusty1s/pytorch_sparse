copy "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Enterprise\\VC\\Tools\\MSVC\\14.29.30037\\lib\\x64\\metis.lib" %LIBRARY_LIB%
copy "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Enterprise\\VC\\Tools\\MSVC\\14.29.30037\\include\\metis.h" %LIBRARY_INC%

:: pip install .
python setup.py install
