#include "macros.h"
#include <torch/extension.h>

// for getpid()
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
