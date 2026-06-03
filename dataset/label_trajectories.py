import argparse
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk

DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = DEFAULT_DATASET_DIR
JSON_DIR = DATASET_DIR / "jsons"
TEXT_DIR = DATASET_DIR / "texts"


def trajectory_index(path):
    match = re.fullmatch(r"trajectory_(\d+)\.json", path.name)
    if not match:
        return None
    return int(match.group(1))


def text_path_for(json_path):
    return TEXT_DIR / f"{json_path.stem}.txt"


def configure_paths(dataset_dir):
    global DATASET_DIR, JSON_DIR, TEXT_DIR

    DATASET_DIR = Path(dataset_dir).expanduser().resolve()
    JSON_DIR = DATASET_DIR / "jsons"
    TEXT_DIR = DATASET_DIR / "texts"


def load_points(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_text(txt_path):
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8-sig").rstrip("\n")


def draw_points(ax, points):
    xs, ys = [], []

    def flush():
        nonlocal xs, ys
        if len(xs) == 1:
            ax.plot(xs, ys, "o", markersize=1, color="black")
        elif len(xs) > 1:
            ax.plot(xs, ys, "k-", linewidth=1)
        xs, ys = [], []

    for x, y, state in points:
        if state == 0:
            xs.append(x)
            ys.append(y)
        elif state == 1:
            flush()

    flush()
    ax.invert_yaxis()
    ax.axis("equal")
    ax.grid(True, linewidth=0.3, alpha=0.25)


class TrajectoryLabeler:
    def __init__(self, json_paths, start_index=0, review=True):
        self.json_paths = json_paths
        self.current = start_index
        self.review = review
        self.updating = False

        self.root = tk.Tk()
        self.root.title("Trajectory Labeler")
        self.root.geometry("1200x850")
        self.root.minsize(900, 650)

        self.go_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.mode_var = tk.StringVar()

        self.build_ui()
        self.bind_shortcuts()
        self.update_mode()
        self.show_current()

    def build_ui(self):
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=(8, 6))
        controls.grid(row=0, column=0, sticky="ew")

        self.prev_button = ttk.Button(controls, text="Prev", command=self.prev_file)
        self.prev_button.pack(side="left")

        self.next_button = ttk.Button(controls, text="Next", command=self.next_file)
        self.next_button.pack(side="left", padx=(6, 14))

        self.mode_button = ttk.Button(controls, textvariable=self.mode_var, command=self.toggle_review)
        self.mode_button.pack(side="left")

        ttk.Label(controls, text="Go #:").pack(side="left", padx=(16, 4))
        self.go_entry = ttk.Entry(controls, textvariable=self.go_var, width=8)
        self.go_entry.pack(side="left")
        self.go_entry.bind("<Return>", lambda event: self.go_to_file())

        self.go_button = ttk.Button(controls, text="Go", command=self.go_to_file)
        self.go_button.pack(side="left", padx=(4, 14))

        self.save_button = ttk.Button(controls, text="Save", command=self.save)
        self.save_button.pack(side="left")

        self.save_next_button = ttk.Button(controls, text="Save+Next", command=self.save_and_next)
        self.save_next_button.pack(side="left", padx=(6, 0))

        self.paste_button = ttk.Button(controls, text="Paste", command=self.paste_clipboard)
        self.paste_button.pack(side="left", padx=(6, 0))

        ttk.Label(
            controls,
            text="Text: Enter = new line, Ctrl+S = save, Ctrl+Enter = save+next",
        ).pack(side="right")

        content = ttk.PanedWindow(self.root, orient="vertical")
        content.grid(row=1, column=0, sticky="nsew")

        plot_frame = ttk.Frame(content)
        text_frame = ttk.Frame(content, padding=(8, 6))
        content.add(plot_frame, weight=4)
        content.add(text_frame, weight=1)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.text = tk.Text(text_frame, height=5, wrap="word", undo=True)
        self.text.grid(row=0, column=0, sticky="nsew")
        self.text.bind("<Control-v>", self.paste_event)
        self.text.bind("<Control-V>", self.paste_event)
        self.text.bind("<Shift-Insert>", self.paste_event)
        self.text.bind("<Control-KeyPress>", self.control_key_event)
        self.text.bind("<Button-3>", self.show_context_menu)

        self.context_menu = tk.Menu(self.root, tearoff=False)
        self.context_menu.add_command(label="Paste", command=self.paste_clipboard)

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scrollbar.set)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(8, 4))
        status.grid(row=2, column=0, sticky="ew")

    def bind_shortcuts(self):
        self.root.bind_all("<Control-s>", lambda event: self.save_event())
        self.root.bind_all("<Control-S>", lambda event: self.save_event())
        self.root.bind_all("<Control-Return>", lambda event: self.save_next_event())
        self.root.bind_all("<Alt-Right>", lambda event: self.next_event())
        self.root.bind_all("<Alt-Left>", lambda event: self.prev_event())
        self.root.bind_all("<Prior>", lambda event: self.prev_event())
        self.root.bind_all("<Next>", lambda event: self.next_event())

    def set_status(self, message):
        self.status_var.set(message)

    def set_text(self, value):
        self.updating = True
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)
        if self.review:
            self.text.configure(state="disabled")
        self.updating = False

    def get_text(self):
        return self.text.get("1.0", "end-1c")

    def update_mode(self):
        mode = "Review" if self.review else "Edit"
        self.mode_var.set(f"Mode: {mode}")
        state = "disabled" if self.review else "normal"
        self.save_button.configure(state=state)
        self.save_next_button.configure(state=state)
        self.paste_button.configure(state=state)
        self.text.configure(state="disabled" if self.review else "normal")

    def toggle_review(self):
        self.review = not self.review
        self.update_mode()
        self.show_current()

    def show_current(self):
        json_path = self.json_paths[self.current]
        txt_path = text_path_for(json_path)
        json_idx = trajectory_index(json_path)
        points = load_points(json_path)
        label_state = "labeled" if txt_path.exists() else "missing txt"

        self.ax.clear()
        draw_points(self.ax, points)
        self.ax.set_title(f"{json_path.name} ({self.current + 1}/{len(self.json_paths)}) - {label_state}")
        self.canvas.draw_idle()

        self.go_var.set(str(json_idx))
        self.set_text(load_label_text(txt_path))

        mode_hint = "read-only" if self.review else "editing"
        self.set_status(f"{json_path.name}: {label_state}, {mode_hint}. Alt+Left/Alt+Right or PgUp/PgDn navigates.")

    def save(self):
        if self.review:
            self.set_status("Review mode is read-only. Click Mode: Review to switch to Edit.")
            return False

        json_path = self.json_paths[self.current]
        txt_path = text_path_for(json_path)
        text = self.get_text()
        if not text.strip():
            self.set_status(f"{json_path.name}: empty text was not saved.")
            return False

        TEXT_DIR.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w", encoding="utf-8", newline="") as f:
            f.write(text)

        self.set_status(f"Saved {txt_path.name}.")
        return True

    def paste_clipboard(self):
        if self.review:
            self.set_status("Review mode is read-only. Click Mode: Review to switch to Edit.")
            return False

        try:
            value = self.root.clipboard_get()
        except tk.TclError:
            self.set_status("Clipboard is empty or does not contain text.")
            return False

        if not value:
            self.set_status("Clipboard is empty.")
            return False

        self.text.insert("insert", value)
        self.set_status(f"Pasted {len(value)} characters.")
        return True

    def save_and_next(self):
        if self.save():
            self.next_file()

    def prev_file(self):
        self.current = (self.current - 1) % len(self.json_paths)
        self.show_current()

    def next_file(self):
        self.current = (self.current + 1) % len(self.json_paths)
        self.show_current()

    def go_to_file(self):
        try:
            requested = int(self.go_var.get().strip())
        except ValueError:
            self.set_status(f"Invalid trajectory number: {self.go_var.get()!r}")
            return

        for i, path in enumerate(self.json_paths):
            if trajectory_index(path) == requested:
                self.current = i
                self.show_current()
                return

        self.set_status(f"trajectory_{requested}.json was not found.")

    def save_event(self):
        self.save()
        return "break"

    def save_next_event(self):
        self.save_and_next()
        return "break"

    def paste_event(self, event=None):
        self.paste_clipboard()
        return "break"

    def control_key_event(self, event):
        # On non-English keyboard layouts Tk may not report keysym as "v".
        # Windows virtual-key code for V is 86, so this catches Ctrl+V by key position too.
        if event.keycode == 86 or event.keysym.lower() == "v":
            return self.paste_event(event)
        return None

    def show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
        return "break"

    def prev_event(self):
        self.prev_file()
        return "break"

    def next_event(self):
        self.next_file()
        return "break"

    def run(self):
        self.root.mainloop()


def find_json_paths():
    paths = []
    for path in JSON_DIR.glob("trajectory_*.json"):
        idx = trajectory_index(path)
        if idx is not None:
            paths.append((idx, path))
    return [path for _, path in sorted(paths)]


def find_start_index(paths, requested_start=None, first_missing=True):
    if requested_start is not None:
        for i, path in enumerate(paths):
            if trajectory_index(path) == requested_start:
                return i
        raise ValueError(f"trajectory_{requested_start}.json was not found")

    if first_missing:
        for i, path in enumerate(paths):
            if not text_path_for(path).exists():
                return i
    return 0


def main():
    parser = argparse.ArgumentParser(description="Label trajectory JSON files with text files.")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Folder that contains jsons/ and texts/. Defaults to the parent of this scripts/ folder.",
    )
    parser.add_argument("--start", type=int, help="Start from trajectory_N.json.")
    parser.add_argument(
        "--review",
        action="store_true",
        help="Start in read-only review mode. This is the default.",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Start in edit mode.",
    )
    args = parser.parse_args()
    configure_paths(args.dataset_dir)

    paths = find_json_paths()
    if not paths:
        raise FileNotFoundError(f"No trajectory_*.json files found in {JSON_DIR}")

    start_in_review = not args.edit or args.review
    start_index = find_start_index(paths, args.start, first_missing=not start_in_review)
    TrajectoryLabeler(paths, start_index=start_index, review=start_in_review).run()


if __name__ == "__main__":
    main()
