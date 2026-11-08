"""Cross-platform process/runtime checks; no GPU/model imports."""
import os
import platform


def supported_host(system=None,machine=None):
    return (system or platform.system()) in ('Windows','Linux') and (machine or platform.machine()).lower() in ('amd64','x86_64')


def process_alive(pid):
    if not isinstance(pid,int) or pid<=0:
        return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype=wintypes.BOOL
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            error=ctypes.get_last_error()
            if error==87:  # nonexistent PID
                return False
            if error==5:  # access denied: conservatively assume alive
                return True
            raise OSError(error,'Unable to query Worker PID')
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value==259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
