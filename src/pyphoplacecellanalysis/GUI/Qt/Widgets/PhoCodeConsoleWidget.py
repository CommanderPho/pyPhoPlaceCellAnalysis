# PhoCodeConsoleWidget

import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

from pyphoplacecellanalysis.External.pyqtgraph.console import ConsoleWidget


class PhoCodeConsoleWidget(ConsoleWidget):
    """
    Widget displaying console output and accepting command input.
    Extends the pyqtgraph ConsoleWidget with terminal-like script execution:

      - ``%run <path> [args...]`` / ``run <path> [args...]`` runs a ``.py`` file via runpy
        (``__name__ == '__main__'``, ``__file__``, ``sys.argv``), then merges public names
        into the interactive namespace. The bare ``run`` form is ignored when the next token is
        ``=`` so assignments like ``run = 1`` still execute as Python.
      - ``!<command>`` runs a shell subprocess when ``enable_shell_commands`` is True.
        On Windows this uses ``shell=True`` (user-controlled command string).

    Shell passthrough can be disabled (e.g. locked-down hosts) by passing
    ``enable_shell_commands=False``.

    Usage:

      import os
      import shlex
      import sys
      import tempfile
      from pathlib import Path

      os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

      from PyQt5 import QtWidgets, uic

      app = QtWidgets.QApplication(sys.argv)

      from pyphoplacecellanalysis.GUI.Qt.Widgets.PhoCodeConsoleWidget import PhoCodeConsoleWidget

      td = tempfile.mkdtemp()
      scr = Path(td) / "hello_console.py"
      scr.write_text(
          "import sys\n"
          "foo = 42\n"
          "if __name__ == '__main__':\n"
          "    main_val = len(sys.argv)\n"
      )

      w = PhoCodeConsoleWidget()
      import runpy
      _dbg = runpy.run_path(str(scr.resolve()), init_globals={}, run_name="__main__")
      print("runpy_debug", _dbg.get("foo"), _dbg.get("main_val"))
      _cmd = '%%run "%s" a b' % scr.as_posix()
      print("cmd repr", repr(_cmd))
      print("shlex", shlex.split(_cmd, posix=(os.name != "nt")))
      w.execSingle(_cmd)
      print("after exec ns keys", "foo" in w.localNamespace, list(w.localNamespace.keys())[:8])
      assert w.localNamespace.get("foo") == 42
      assert w.localNamespace.get("main_val") == 3, w.localNamespace.get("main_val")

      w2 = PhoCodeConsoleWidget(enable_shell_commands=False)
      w2.execSingle("!echo should_not_run")

      w3 = PhoCodeConsoleWidget(enable_shell_commands=True)
      w3.execSingle("!echo pho_smoke_ok")

      # UI load (MainPipelineWindow.ui)
      here = Path(__file__).resolve().parent
      ui = here / "src" / "pyphoplacecellanalysis" / "GUI" / "PyQtPlot" / "Windows" / "MainPipelineWindow.ui"
      mw = uic.loadUi(str(ui), QtWidgets.QMainWindow())
      console = mw.findChild(PhoCodeConsoleWidget, "console")
      assert console is not None, type(mw.findChild(QtWidgets.QWidget, "console"))

      print("smoke_ok", td)

    """

    _RUN_MERGE_BLOCKLIST = frozenset({
        '__builtins__', '__console__', '__loader__', '__spec__', '__package__',
        '__name__', '__file__', '__doc__', '__cached__',
    })


    def __init__(self, parent=None, namespace=None, historyFile=None, text=None, editor=None, enable_shell_commands=True):
        self._enable_shell_commands = enable_shell_commands
        super(PhoCodeConsoleWidget, self).__init__(parent=parent, namespace=namespace, historyFile=historyFile, text=text, editor=editor)


    def execSingle(self, cmd):
        stripped = cmd.strip()
        if stripped.startswith('!'):
            if not self._enable_shell_commands:
                self.write('Shell commands (!) are disabled for this console.\n')
                return
            self._run_shell_line(stripped[1:].strip())
            return
        if self._try_handle_run_magic(stripped):
            return
        super(PhoCodeConsoleWidget, self).execSingle(cmd)


    def _try_handle_run_magic(self, stripped):
        if not stripped:
            return False
        try:
            parts = shlex.split(stripped, posix=(os.name != 'nt'))
        except ValueError:
            self.write('Could not parse command (check quoting).\n')
            return True
        if not parts:
            return False
        if parts[0] == '%run':
            if len(parts) < 2:
                self.write('%run requires a script path.\n')
                return True
            self._run_script_path(parts[1], parts[2:])
            return True
        if parts[0] == 'run':
            if len(parts) < 2:
                return False
            if parts[1] == '=':
                return False
            self._run_script_path(parts[1], parts[2:])
            return True
        return False


    def _run_script_path(self, script_path_s, argv_rest):
        try:
            s = script_path_s.strip()
            if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                s = s[1:-1]
            path = Path(s)
            if not path.is_absolute():
                path = Path(os.getcwd()) / path
            path = path.resolve()
            if not path.is_file():
                self.write('Not a file: %s\n' % (path,))
                return
            init_globals = dict(self.globals())
            old_argv = sys.argv
            sys.argv = [str(path)] + list(argv_rest)
            try:
                run_globals = runpy.run_path(str(path), init_globals=init_globals, run_name='__main__')
            finally:
                sys.argv = old_argv
            for key, val in run_globals.items():
                if key not in self._RUN_MERGE_BLOCKLIST:
                    self.localNamespace[key] = val
        except Exception:
            self.displayException()


    def _run_shell_line(self, cmd_line):
        if not cmd_line:
            self.write('Empty shell command.\n')
            return
        try:
            use_shell = sys.platform == 'win32'
            completed = subprocess.run(cmd_line, shell=use_shell, capture_output=True, text=True, cwd=os.getcwd())
            if completed.stdout:
                self.write(completed.stdout)
            if completed.stderr:
                self.write(completed.stderr)
            if completed.returncode != 0:
                self.write('\n[exit code %s]\n' % (completed.returncode,))
        except Exception:
            self.displayException()
