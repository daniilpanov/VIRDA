import os
import sys
import traceback

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ttk_inside_pyi = os.path.join(sys._MEIPASS, "ttkbootstrap")
    if os.path.exists(ttk_inside_pyi):
        sys.path.insert(0, sys._MEIPASS)

if __name__ == "__main__":
    from virda_gui.app import main

    # In a windowed (console=False) frozen build stderr is lost, so a
    # traceback from startup would vanish silently. Route fatal errors to a
    # crash log next to the executable before exiting non-zero.
    if getattr(sys, "frozen", False):

        def _frozen_excepthook(exc_type, exc_value, exc_tb) -> None:
            crash_log = os.path.join(os.path.dirname(sys.executable), "virda-crash.log")
            try:
                with open(crash_log, "w", encoding="utf-8") as fh:
                    traceback.print_exception(exc_type, exc_value, exc_tb, file=fh)
            except OSError:
                pass
            traceback.print_exception(exc_type, exc_value, exc_tb)
            sys.exit(1)

        sys.excepthook = _frozen_excepthook

    main()
