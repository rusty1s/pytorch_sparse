try:
    import pyg_lib  # noqa
    WITH_PYG_LIB = True
    WITH_INDEX_SORT = hasattr(pyg_lib.ops, 'index_sort')
except ImportError:
    pyg_lib = object
    WITH_PYG_LIB = False
    WITH_INDEX_SORT = False
