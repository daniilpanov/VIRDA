import os
import sys

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ttk_inside_pyi = os.path.join(sys._MEIPASS, "ttkbootstrap")
    if os.path.exists(ttk_inside_pyi):
        sys.path.insert(0, sys._MEIPASS)

if __name__ == "__main__":
    from virda_gui.app import main

    main()
