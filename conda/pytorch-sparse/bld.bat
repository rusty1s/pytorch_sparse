copy "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Enterprise\\VC\\Tools\\MSVC\\14.29.30133\\lib\\x64\\metis.lib" %LIBRARY_LIB%
if errorlevel 1 exit 1
copy "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Enterprise\\VC\\Tools\\MSVC\\14.29.30133\\include\\metis.h" %LIBRARY_INC%
if errorlevel 1 exit 1

"%PYTHON%" -m pip install .
if errorlevel 1 exit 1
