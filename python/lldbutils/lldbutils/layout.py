import lldb

def frametree(debugger, command, result, dict):
    """Dumps the frame tree containing the given nsIFrame*."""
    debugger.HandleCommand('expr (' + command + ')->DumpFrameTree()')

def frametreelimited(debugger, command, result, dict):
    """Dumps the subtree of a frame tree rooted at the given nsIFrame*."""
    debugger.HandleCommand('expr (' + command + ')->DumpFrameTreeLimited()')

def init(debugger):
    debugger.HandleCommand('command script add -f lldbutils.layout.frametree frametree')
    debugger.HandleCommand('command script add -f lldbutils.layout.frametreelimited frametreelimited')
    debugger.HandleCommand('command alias ft frametree')
    debugger.HandleCommand('command alias ftl frametreelimited')
