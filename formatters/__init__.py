import logging
from . import lean

log = logging.getLogger(__name__)


def __lldb_init_module(debugger_obj, internal_dict):  # pyright: ignore
    log.info('Initializing')
    lean.__lldb_init_module(debugger_obj, internal_dict)