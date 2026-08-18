"""Reusable tkinter widgets for the VIRDA GUI application.

Provides ``FileSelector``, ``DirectorySelector``, ``CollapsibleSection`` and
``LogViewer`` — thin wrappers around standard ttk / ttkbootstrap widgets that
reduce boilerplate in the main application window.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.scrolledtext as scrolledtext
from datetime import datetime
from tkinter import ttk
from typing import Literal


class FileSelector(ttk.Frame):
    """A label + entry + *Browse* button row for selecting a file."""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        filetypes: list[tuple[str, str]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(master, **kwargs)
        self._filetypes = filetypes or [("All files", "*.*")]

        self._label = ttk.Label(self, text=label, width=14, anchor="w")
        self._label.pack(side=tk.LEFT, padx=(0, 6))

        self._var = tk.StringVar()
        self._entry = ttk.Entry(self, textvariable=self._var, width=48)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._btn = ttk.Button(self, text="Browse", command=self._browse, width=8)
        self._btn.pack(side=tk.LEFT, padx=(6, 0))

    def _browse(self) -> None:
        path = filedialog.askopenfilename(filetypes=self._filetypes)
        if path:
            self._var.set(path)

    def get(self) -> str:
        """Return the current path value (empty string if unset)."""
        return self._var.get()

    def set(self, value: str) -> None:
        self._var.set(value)


class DirectorySelector(ttk.Frame):
    """A label + entry + *Browse* button row for selecting a directory."""

    def __init__(self, master: tk.Misc, label: str, **kwargs: object) -> None:
        super().__init__(master, **kwargs)

        self._label = ttk.Label(self, text=label, width=14, anchor="w")
        self._label.pack(side=tk.LEFT, padx=(0, 6))

        self._var = tk.StringVar()
        self._entry = ttk.Entry(self, textvariable=self._var, width=48)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._btn = ttk.Button(self, text="Browse", command=self._browse, width=8)
        self._btn.pack(side=tk.LEFT, padx=(6, 0))

    def _browse(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self._var.set(path)

    def get(self) -> str:
        return self._var.get()

    def set(self, value: str) -> None:
        self._var.set(value)


class CollapsibleSection(ttk.LabelFrame):
    """A ``LabelFrame`` that can be expanded / collapsed by clicking its title.

    Child widgets added via the *body* frame are shown or hidden on toggle.
    """

    def __init__(self, master: tk.Misc, title: str, **kwargs: object) -> None:
        super().__init__(master, text=f"  {title}", **kwargs)
        self._expanded = True

        self._body = ttk.Frame(self)
        self._body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.bind("<Button-1>", self._toggle)
        self._body.bind("<Button-1>", self._toggle)

    @property
    def body(self) -> ttk.Frame:
        """The inner frame where callers place child widgets."""
        return self._body

    def _toggle(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        if self._expanded:
            self._body.pack_forget()
            self._expanded = False
        else:
            self._body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
            self._expanded = True

    def collapse(self) -> None:
        if self._expanded:
            self._toggle()

    def expand(self) -> None:
        if not self._expanded:
            self._toggle()


class LabeledField(ttk.Frame):
    """A compact label + widget row inside a ``CollapsibleSection``."""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        widget_type: Literal["entry", "combo", "check"] = "entry",
        values: list[str] | None = None,
        default: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(master, **kwargs)

        self._label = ttk.Label(self, text=label, width=20, anchor="w")
        self._label.pack(side=tk.LEFT, padx=(0, 6))

        self._var = tk.StringVar(value=default or "")

        if widget_type == "combo":
            self._widget = ttk.Combobox(
                self, textvariable=self._var, values=values or [], state="readonly", width=16
            )
            self._widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
        elif widget_type == "check":
            self._widget = ttk.Checkbutton(
                self, variable=self._var, onvalue="true", offvalue="false"
            )
            self._widget.pack(side=tk.LEFT)
        else:
            self._widget = ttk.Entry(self, textvariable=self._var, width=16)
            self._widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def get(self) -> str:
        return self._var.get()

    def set(self, value: str) -> None:
        self._var.set(value)


class LogViewer(ttk.LabelFrame):
    """A read-only scrolled text area for displaying pipeline log output."""

    def __init__(self, master: tk.Misc, **kwargs: object) -> None:
        super().__init__(master, text="Log", **kwargs)

        self._text = scrolledtext.ScrolledText(
            self, height=12, state=tk.DISABLED, wrap=tk.WORD, font=("TkDefaultFont", 9)
        )
        self._text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def append(self, message: str) -> None:
        """Append a timestamped line to the log."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._text.configure(state=tk.NORMAL)
        self._text.insert(tk.END, f"[{ts}] {message}\n")
        self._text.see(tk.END)
        self._text.configure(state=tk.DISABLED)

    def clear(self) -> None:
        """Clear all log content."""
        self._text.configure(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.configure(state=tk.DISABLED)
