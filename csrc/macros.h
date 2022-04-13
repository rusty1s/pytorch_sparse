#pragma once

#ifdef _WIN32
#if defined(torchsparse_EXPORTS)
#define SPARSE_API __declspec(dllexport)
#else
#define SPARSE_API __declspec(dllimport)
#endif
#else
#define SPARSE_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define SPARSE_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define SPARSE_INLINE_VARIABLE __declspec(selectany)
#else
#define SPARSE_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
