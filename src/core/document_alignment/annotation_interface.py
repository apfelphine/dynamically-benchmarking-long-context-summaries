import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, simpledialog
import json
import uuid
import time
import sys
import numpy as np

# DPI Awareness for crisp UI on Windows
if sys.platform == "win32":
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)  # noqa
    except Exception as e:
        print(f"DPI Awareness Error: {e}")

COLORS = [
    "#FFD700",
    "#FF69B4",
    "#7FFF00",
    "#00FFFF",  # noqa
    "#FFA07A",
    "#BA55D3",
    "#87CEFA",  # noqa
    "#F08080",
]


def lighten_color(hex_color, factor):
    """
    Blends the given hex color with white.
    factor: 0.0 (white) -> 1.0 (original color)
    """
    # Clamp factor to ensure visibility (min 0.15)
    factor = max(0.15, min(1.0, factor))

    if not hex_color.startswith("#"):
        return hex_color

    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Blend with white (255)
    r_new = int(r * factor + 255 * (1 - factor))
    g_new = int(g * factor + 255 * (1 - factor))
    b_new = int(b * factor + 255 * (1 - factor))

    return f"#{r_new:02x}{g_new:02x}{b_new:02x}"


class SearchDialog(tk.Toplevel):
    def __init__(self, master, text_widget, text, **kwargs):
        super().__init__(master, **kwargs)
        self.text_widget = text_widget
        self.title(f"Find Text in {text}")
        self.transient(master)
        self.resizable(False, False)

        self.match_indices = []
        self.current_match_index = -1

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)

        ttk.Label(main_frame, text="Find:").grid(row=0, column=0, sticky="w")

        self.search_entry = ttk.Entry(main_frame)
        self.search_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        self.search_entry.focus_set()

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky="e", pady=(5, 0))

        ttk.Button(button_frame, text="Previous", command=self.find_previous).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Next", command=self.find_next).pack(side="left")

        self.status_label = ttk.Label(main_frame, text="0 matches found")
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Event bindings
        self.search_entry.bind("<Return>", lambda _: self.find_next())
        self.search_entry.bind("<KeyRelease>", lambda _: self.on_entry_change())
        self.bind("<Escape>", lambda _: self.close_dialog())

        # Tag configs
        self.text_widget.tag_config("search_match", background="#a5d6a7")
        self.text_widget.tag_config("search_match_selected", background="#66bb6a")

        self.protocol("WM_DELETE_WINDOW", self.close_dialog)

    def highlight_all_matches(self):
        self.text_widget.tag_remove("search_match", "1.0", "end")
        self.text_widget.tag_remove("search_match_selected", "1.0", "end")
        self.match_indices.clear()
        self.current_match_index = -1

        search_term = self.search_entry.get()
        if not search_term:
            self.status_label.config(text="0 matches found")
            return

        pos = "1.0"
        while True:
            pos = self.text_widget.search(
                search_term, pos, stopindex="end", nocase=True
            )
            if not pos:
                break
            end_pos = f"{pos}+{len(search_term)}c"
            self.match_indices.append((pos, end_pos))
            self.text_widget.tag_add("search_match", pos, end_pos)
            pos = end_pos

        total = len(self.match_indices)
        self.status_label.config(
            text=f"{total} match{'es' if total != 1 else ''} found"
        )

    def find_next(self):
        if not self.match_indices:
            self.highlight_all_matches()
        if not self.match_indices:
            return

        self.current_match_index = (self.current_match_index + 1) % len(
            self.match_indices
        )
        self.select_current_match()

    def find_previous(self):
        if not self.match_indices:
            self.highlight_all_matches()
        if not self.match_indices:
            return

        self.current_match_index = (self.current_match_index - 1) % len(
            self.match_indices
        )
        self.select_current_match()

    def select_current_match(self):
        self.text_widget.tag_remove("search_match_selected", "1.0", "end")

        start, end = self.match_indices[self.current_match_index]
        self.text_widget.tag_add("search_match_selected", start, end)
        self.text_widget.see(start)
        self.text_widget.mark_set(tk.INSERT, end)
        self.text_widget.focus_set()

        self.status_label.config(
            text=f"{self.current_match_index + 1} of {len(self.match_indices)} matches"
        )

    def on_entry_change(self):
        self.highlight_all_matches()

    def close_dialog(self):
        self.text_widget.tag_remove("search_match", "1.0", "end")
        self.text_widget.tag_remove("search_match_selected", "1.0", "end")
        self.destroy()
        self.master.search_dialog = None


class AnnotationTool(tk.Tk):
    def __init__(self):
        super().__init__()

        # Initialize ui elements
        self.selection_frame = None
        self.sidebar_list_frame = None
        self.canvas_frame_id = None
        self.sidebar_canvas = None
        self.sum_text = None
        self.confirm_button = None
        self.stage_button = None
        self.clear_staged_button = None
        self.sum_selections_text = None
        self.doc_selections_text = None
        self.status_label = None
        self.view_menu = None
        self.file_menu = None
        self.doc_notebook = None
        self.doc_notebook_frame = None
        self.current_project_path = None
        self.search_dialog = None

        self.title("Source-Summary Annotation Tool")
        self.geometry("1400x800")
        self.state("zoomed")

        self.next_color_index = 0

        self.annotations = []
        self.staged_doc_selections = []
        self.staged_sum_selections = []
        self.staging_history = []

        self.view_selected_only_var = tk.BooleanVar(value=False)
        self.display_threshold = 0.0  # Show all by default
        self.selected_annotation_id = None

        self.view_only_var = tk.BooleanVar(value=False)

        self.doc_text_widgets = []
        self.autosave_enabled = tk.BooleanVar(value=True)

        self.show_scores = False

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Configure the main button style
        style.configure(
            "Accent.TButton",
            foreground="white",
            background="#3b82f6",
            font=("Helvetica", 12, "bold"),
        )
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure("Vertical.TNotebook", tabposition="wn")

        # Configure the Tab (Document List Item) appearance
        style.configure(
            "Vertical.TNotebook.Tab",
            padding=[10, 8],
            font=("Helvetica", 10),
            anchor="w",
            width=20,
        )

        style.map(
            "Vertical.TNotebook.Tab",
            background=[("selected", "#e3f2fd")],
            foreground=[("selected", "#000000")],
        )

        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # Open / Load Project Menu
        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(
            label="New Project from document & summary files...",
            command=self.new_project_from_txt,
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Load Project from Similarity Matrix...",
            command=self.new_project_from_sim_matrix,
        )
        self.file_menu.add_command(label="Load Project...", command=self.load_project)
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Save", command=self.save_project, state="disabled"
        )
        self.file_menu.add_command(
            label="Save Project As...", command=self.save_project_as
        )
        self.file_menu.add_separator()
        self.file_menu.add_checkbutton(
            label="Auto-Save Enabled",
            onvalue=True,
            offvalue=False,
            variable=self.autosave_enabled,
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.destroy)

        # View Menu
        self.view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_checkbutton(
            label="Read-Only Mode",
            onvalue=True,
            offvalue=False,
            variable=self.view_only_var,
            command=self.toggle_view_only,
        )
        self.view_menu.add_separator()
        self.view_menu.add_checkbutton(
            label="View only selected annotation",
            onvalue=True,
            offvalue=False,
            variable=self.view_selected_only_var,
            command=self.apply_visual_styles,
        )
        self.view_menu.add_separator()
        self.view_menu.add_command(
            label="Set Display Threshold...", command=self.change_display_threshold
        )

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(2, weight=2)
        main_frame.rowconfigure(1, weight=1)

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky="ew")

        self.status_label = ttk.Label(
            control_frame, text="Welcome! Load a project or create a new one to begin."
        )
        self.status_label.pack(side="right", padx=10)

        selection_frame = ttk.Frame(main_frame)
        self.selection_frame = selection_frame
        selection_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky="ew")
        selection_frame.columnconfigure(0, weight=1)
        selection_frame.columnconfigure(1, weight=1)
        ttk.Label(selection_frame, text="Document Selections:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(selection_frame, text="Summary Selections:").grid(
            row=0, column=1, sticky="w"
        )
        self.doc_selections_text = scrolledtext.ScrolledText(
            selection_frame, height=3, wrap="word", state="disabled"
        )
        self.doc_selections_text.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.sum_selections_text = scrolledtext.ScrolledText(
            selection_frame, height=3, wrap="word", state="disabled"
        )
        self.sum_selections_text.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        action_button_frame = ttk.Frame(selection_frame)
        action_button_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        action_button_frame.columnconfigure(0, weight=1)
        action_button_frame.columnconfigure(1, weight=1)
        action_button_frame.columnconfigure(2, weight=1)
        self.stage_button = ttk.Button(
            action_button_frame,
            text="Stage Selection",
            command=self.smart_stage_selection,
        )
        self.stage_button.grid(row=0, column=0, sticky="ew", padx=5)
        self.clear_staged_button = ttk.Button(
            action_button_frame, text="Clear All Staged", command=self.clear_all_staged
        )
        self.clear_staged_button.grid(row=0, column=1, sticky="ew", padx=5)
        self.confirm_button = ttk.Button(
            action_button_frame,
            text="✅ Confirm Correlation",
            command=self.confirm_correlation,
            style="Accent.TButton",
        )
        self.confirm_button.grid(row=0, column=2, sticky="ew", padx=5)

        # Document Notebook (for tabs for different documents)
        self.doc_notebook_frame = ttk.LabelFrame(main_frame, text="Source Document(s)")
        self.doc_notebook_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.doc_notebook_frame.columnconfigure(0, weight=1)
        self.doc_notebook_frame.rowconfigure(0, weight=1)

        # Use the "Vertical" style we defined at the top
        self.doc_notebook = ttk.Notebook(
            self.doc_notebook_frame, style="Vertical.TNotebook"
        )

        self.doc_notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.doc_text_widgets = []
        self.doc_notebook.bind("<<NotebookTabChanged>>", self.on_doc_tab_changed)

        self.sum_text = self.configure_text_box(
            main_frame, index=1, title="Summary Text"
        )

        self.setup_sidebar(main_frame)
        self.configure_tags()

        self.bind("<Control-s>", lambda _e: self.on_save())

        def on_c_press(event):
            if (event.state & 0x0004) != 0:
                return
            self.confirm_correlation()

        self.bind("<c>", lambda _e: on_c_press(_e))
        self.bind("<x>", lambda _e: self.clear_all_staged())
        self.bind("<Control-z>", lambda _e: self.undo_last_action())

    def configure_tags(self):
        for widget in [self.sum_text]:
            widget.tag_config("staged", background="#b3e5fc")
            widget.tag_config("search_match", background="#a5d6a7")

    def create_doc_text_widget(self, parent_notebook):
        """Helper to create a new tab with a ScrolledText widget for the doc notebook"""
        text_frame = ttk.Frame(parent_notebook)  # Frame for the tab content
        text_frame.pack(fill="both", expand=True)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        scrolled_text = scrolledtext.ScrolledText(
            text_frame, wrap="word", font=("Helvetica", 12)
        )
        scrolled_text.grid(
            row=0, column=0, sticky="nsew", padx=0, pady=0
        )  # No internal padding
        scrolled_text.config(state="disabled")

        scrolled_text.bind(
            "<Control-f>", lambda _e: self.open_search_dialog(_e, "Document")
        )
        scrolled_text.bind("<a>", self.handle_stage)

        # Configure tags for this new widget
        scrolled_text.tag_config("staged", background="#b3e5fc")
        scrolled_text.tag_config("search_match", background="#a5d6a7")

        return scrolled_text, text_frame

    def configure_text_box(self, main_frame, index, title):
        """Configures the summary text box"""
        text_frame = ttk.LabelFrame(main_frame, text=title)
        text_frame.grid(row=1, column=index, padx=10, pady=5, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        scrolled_text = scrolledtext.ScrolledText(
            text_frame, wrap="word", font=("Helvetica", 12)
        )
        scrolled_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        scrolled_text.config(state="disabled")

        scrolled_text.bind("<Control-f>", lambda _e: self.open_search_dialog(_e, title))
        scrolled_text.bind("<a>", self.handle_stage)

        return scrolled_text

    def setup_sidebar(self, parent_frame):
        sidebar_frame = ttk.LabelFrame(parent_frame, text="Correlations List")
        sidebar_frame.grid(row=1, column=2, padx=10, pady=5, sticky="nsew")
        sidebar_frame.rowconfigure(0, weight=1)
        sidebar_frame.columnconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(sidebar_frame, borderwidth=0)
        self.sidebar_list_frame = ttk.Frame(self.sidebar_canvas)
        scrollbar = ttk.Scrollbar(
            sidebar_frame, orient="vertical", command=self.sidebar_canvas.yview
        )
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_frame_id = self.sidebar_canvas.create_window(
            (0, 0), window=self.sidebar_list_frame, anchor="n"
        )
        self.sidebar_list_frame.bind("<Configure>", self.on_frame_configure)
        self.sidebar_canvas.bind("<Configure>", self.on_canvas_configure)

    def on_frame_configure(self, _=None):
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def on_canvas_configure(self, event=None):
        self.sidebar_canvas.itemconfig(self.canvas_frame_id, width=event.width)

    def get_current_doc_widget(self):
        """Returns the ScrolledText widget of the currently active document tab"""
        if not self.doc_text_widgets:
            return None
        try:
            selected_tab_id = self.doc_notebook.select()
            return self.doc_text_widgets[self.doc_notebook.index(selected_tab_id)]
        except (tk.TclError, IndexError):
            return None

    def get_current_doc_index(self):
        """Returns the index of the currently active document tab"""
        if not self.doc_text_widgets:
            return -1
        try:
            return self.doc_notebook.index(self.doc_notebook.select())
        except tk.TclError:
            return -1

    def on_doc_tab_changed(self, _):
        """Called when the user switches document tabs."""
        if self.staged_doc_selections:
            self.clear_staged_docs()

    def clear_staged_docs(self, update_status=True):
        """Clears only the document selections from staging."""
        self.staging_history = [
            item for item in self.staging_history if item[0] != "doc"
        ]
        self.staged_doc_selections = []
        self.update_staged_display("doc")
        if update_status:
            self.status_label.config(text="Document staging cleared due to tab switch.")

    def change_display_threshold(self):
        """Dialog to change the confidence threshold for displaying annotations."""
        new_val = simpledialog.askfloat(
            "Set Threshold",
            f"Enter minimum confidence score (0.0 - 1.0):\nCurrent: {self.display_threshold}",
            minvalue=0.0,
            maxvalue=1.0,
            initialvalue=self.display_threshold,
        )
        if new_val is not None:
            self.display_threshold = new_val
            self.update_annotation_list()
            self.apply_visual_styles()
            self.status_label.config(
                text=f"Display threshold set to {int(new_val * 100)}%"
            )

    def apply_visual_styles(self):
        """
        Updates the visual appearance.
        - Summary: Uses aggregated score color.
        - Document: Applies unique individual colors to each chunk based on score.
        - Hides annotations below self.display_threshold.
        """
        only_selected = self.view_selected_only_var.get()
        threshold = self.display_threshold

        for ann in self.annotations:
            tag_name = ann["tag_name"]
            base_color = ann["color"]

            selections = ann["doc_selections"]
            chunk_scores = ann.get("doc_chunk_scores", [1.0] * len(selections))

            # Determine if ANY chunk passes the threshold
            valid_indices = [i for i, s in enumerate(chunk_scores) if s >= threshold]

            # If NO chunks pass the threshold, hide the entire annotation
            if not valid_indices:
                self.sum_text.tag_config(
                    tag_name, background="", relief="flat", borderwidth=0
                )
                doc_idx = ann.get("document_idx", 0)
                if 0 <= doc_idx < len(self.doc_text_widgets):
                    dw = self.doc_text_widgets[doc_idx]
                    dw.tag_config(tag_name, background="", relief="flat", borderwidth=0)
                    for i in range(len(selections)):
                        dw.tag_remove(f"{tag_name}_c{i}", "1.0", "end")
                continue

            is_selected = ann["id"] == self.selected_annotation_id
            valid_scores = [chunk_scores[i] for i in valid_indices]
            effective_score = max(valid_scores) if valid_scores else 0

            # Apply styles to summary
            sum_bg_color = lighten_color(
                base_color, effective_score if self.show_scores else 1.0
            )

            if is_selected:
                relief, border = "solid", 1
            elif only_selected:
                sum_bg_color = ""
                relief, border = "flat", 0
            else:
                relief, border = "flat", 0

            self.sum_text.tag_config(
                tag_name, background=sum_bg_color, relief=relief, borderwidth=border
            )

            # Apply styles to documents
            doc_idx = ann.get("document_idx", 0)
            if 0 <= doc_idx < len(self.doc_text_widgets):
                doc_widget = self.doc_text_widgets[doc_idx]

                doc_widget.tag_config(
                    tag_name, background="", relief=relief, borderwidth=border
                )

                for i, (sel, score) in enumerate(zip(selections, chunk_scores)):
                    chunk_tag = f"{tag_name}_c{i}"

                    should_show = True
                    if score < threshold:
                        should_show = False
                    elif only_selected and not is_selected:
                        should_show = False

                    if should_show:
                        # If show_scores is False, use solid color
                        chunk_bg = lighten_color(
                            base_color, score if self.show_scores else 1.0
                        )

                        doc_widget.tag_config(chunk_tag, background=chunk_bg)
                        doc_widget.tag_add(chunk_tag, sel[0], sel[1])
                        doc_widget.tag_add(tag_name, sel[0], sel[1])
                    else:
                        doc_widget.tag_remove(chunk_tag, sel[0], sel[1])
                        doc_widget.tag_remove(tag_name, sel[0], sel[1])

                doc_widget.tag_raise(tag_name)

    @staticmethod
    def _get_score_color(score_pct: int):
        if score_pct >= 90:
            return "#2e7d32"  # Green
        elif score_pct >= 75:
            return "#1565c0"  # Blue
        elif score_pct >= 50:
            return "#ef6c00"  # Orange
        return "#c62828"  # Red

    def update_annotation_list(self):
        for widget in self.sidebar_list_frame.winfo_children():
            widget.destroy()

        threshold = self.display_threshold

        for ann in sorted(self.annotations, key=lambda x: x.get("created_at", 0)):
            doc_selections = ann["doc_selections"]
            chunk_scores = ann.get("doc_chunk_scores", [1.0] * len(doc_selections))
            valid_indices = [i for i, s in enumerate(chunk_scores) if s >= threshold]

            if not valid_indices:
                continue

            item_frame = ttk.Frame(
                self.sidebar_list_frame, padding=5, relief="groove", borderwidth=1
            )
            item_frame.pack(fill="x", expand=True, pady=2, padx=2)
            item_frame.columnconfigure(1, weight=1)

            color, ann_id = ann["color"], ann["id"]

            valid_scores = [chunk_scores[i] for i in valid_indices]
            agg_score_pct = int(max(valid_scores) * 100) if valid_scores else 0
            agg_score_color = self._get_score_color(agg_score_pct)

            # Color text
            color_label = tk.Label(item_frame, bg=color, width=2, relief="sunken")
            color_label.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 5))

            # Header row
            header_frame = ttk.Frame(item_frame)
            header_frame.grid(row=0, column=1, sticky="ew")

            if self.show_scores:
                tk.Label(
                    header_frame,
                    text=f"Max: {agg_score_pct}%",
                    fg="white",
                    bg=agg_score_color,
                    font=("Helvetica", 7, "bold"),
                    padx=2,
                ).pack(side="right", padx=2)

            try:
                sum_preview = (
                    self.sum_text.get(
                        ann["sum_selections"][0][0], ann["sum_selections"][0][1]
                    )
                    .strip()
                    .replace("\n", " ")
                )
                sum_preview = (
                    (sum_preview[:17] + "...") if len(sum_preview) > 20 else sum_preview
                )
            except:  # noqa
                sum_preview = "Error"

            sum_lbl = ttk.Label(
                header_frame, text=f"Sum: {sum_preview}", font=("Helvetica", 9, "bold")
            )
            sum_lbl.pack(side="left", fill="x")

            # Details row
            doc_container = ttk.Frame(item_frame)
            doc_container.grid(row=1, column=1, sticky="ew", pady=(2, 0))

            doc_selections = ann["doc_selections"]
            doc_scores = ann.get("doc_chunk_scores", [1.0] * len(doc_selections))

            doc_idx = ann.get("document_idx", 0)
            doc_widget = None
            if 0 <= doc_idx < len(self.doc_text_widgets):
                doc_widget = self.doc_text_widgets[doc_idx]

            for i, (sel, score) in enumerate(zip(doc_selections, doc_scores)):
                if score < threshold:
                    continue

                chunk_frame = ttk.Frame(doc_container)
                chunk_frame.pack(fill="x", anchor="w")

                if self.show_scores:
                    score_pct = int(score * 100)
                    sc_col = self._get_score_color(score_pct)
                    tk.Label(
                        chunk_frame,
                        text=f"{score_pct}%",
                        fg=sc_col,
                        font=("Helvetica", 8),
                        width=4,
                        anchor="e",
                    ).pack(side="left", padx=(0, 4))

                if doc_widget:
                    try:
                        txt = doc_widget.get(sel[0], sel[1]).strip().replace("\n", " ")
                        txt = (txt[:17] + "...") if len(txt) > 20 else txt
                    except:  # noqa
                        txt = "Err"
                else:
                    txt = "Invalid Doc"

                chunk_lbl = ttk.Label(
                    chunk_frame, text=txt, font=("Helvetica", 9), cursor="hand2"
                )
                chunk_lbl.pack(side="left", fill="x")
                chunk_lbl.bind(
                    "<Button-1>", lambda _, i=ann_id: self.scroll_to_annotation(i)
                )

            if not self.view_only_var.get():
                delete_button = ttk.Button(
                    item_frame,
                    text="❌",
                    width=3,
                    command=lambda i=ann_id: self.delete_annotation_by_id(i),
                )
                delete_button.grid(row=0, column=2, rowspan=2, sticky="ns", padx=(5, 0))

            for widget in [color_label, item_frame, header_frame, sum_lbl]:
                widget.bind(
                    "<Button-1>", lambda _, i=ann_id: self.scroll_to_annotation(i)
                )

        self.sidebar_list_frame.update_idletasks()
        self.on_frame_configure()

    @staticmethod
    def get_selection_indices(text_widget):
        try:
            return text_widget.index(tk.SEL_FIRST), text_widget.index(tk.SEL_LAST)
        except tk.TclError:
            return None, None

    def handle_stage(self, event):
        if self.view_only_var.get():
            return "break"
        widget = event.widget
        if widget in self.doc_text_widgets:
            self.stage_selection(widget, "doc")
        elif widget == self.sum_text:
            self.stage_selection(self.sum_text, "sum")
        return "break"

    def update_staged_display(self, side):
        if side == "doc":
            text_widget = self.get_current_doc_widget()
            if not text_widget:
                return  # No doc widget active
            s_list = self.staged_doc_selections
            d_widget = self.doc_selections_text
        else:
            text_widget = self.sum_text
            s_list = self.staged_sum_selections
            d_widget = self.sum_selections_text

        text_widget.tag_remove("staged", "1.0", "end")
        for start, end in s_list:
            text_widget.tag_add("staged", start, end)
        d_widget.config(state="normal")
        d_widget.delete("1.0", "end")
        for i, (s, event) in enumerate(s_list):
            d_widget.insert("end", f"[{i + 1}] '{text_widget.get(s, event)[:20]}...'\n")
        d_widget.config(state="disabled")

    def smart_stage_selection(self):
        current_doc_widget = self.get_current_doc_widget()
        doc_sel = (None, None)
        if current_doc_widget:
            doc_sel = self.get_selection_indices(current_doc_widget)
        sum_sel = self.get_selection_indices(self.sum_text)

        if doc_sel[0]:
            self.stage_selection(current_doc_widget, "doc")
        elif sum_sel[0]:
            self.stage_selection(self.sum_text, "sum")
        else:
            messagebox.showwarning("No Selection", "Please highlight text to stage it.")

    def stage_selection(self, text_widget, side):
        start_idx, end_idx = self.get_selection_indices(text_widget)
        if not start_idx:
            return

        # Store the exact index strings from the widget
        start_idx_str = text_widget.index(start_idx)
        end_idx_str = text_widget.index(end_idx)

        self.staging_history.append((side, start_idx_str, end_idx_str))
        (
            self.staged_doc_selections if side == "doc" else self.staged_sum_selections
        ).append((start_idx_str, end_idx_str))

        self.update_staged_display(side)
        self.status_label.config(
            text=f"{len(self.staged_doc_selections) + len(self.staged_sum_selections)} selections staged."
        )
        text_widget.tag_remove(tk.SEL, "1.0", "end")

    def undo_last_action(self):
        if self.view_only_var.get():
            return

        # Undo last staged selection (based on recorded history)
        if self.staging_history:
            last_side, start, end = self.staging_history.pop()

            if last_side == "doc" and self.staged_doc_selections:
                try:
                    self.staged_doc_selections.remove((start, end))
                except ValueError:
                    pass  # In case it was already removed
                self.update_staged_display("doc")
                self.status_label.config(text="Undid last document staging.")
            elif last_side == "sum" and self.staged_sum_selections:
                try:
                    self.staged_sum_selections.remove((start, end))
                except ValueError:
                    pass
                self.update_staged_display("sum")
                self.status_label.config(text="Undid last summary staging.")
            return

        elif self.annotations:
            last_ann = self.annotations.pop()

            self.staged_doc_selections = last_ann.get("doc_selections", [])
            self.staged_sum_selections = last_ann.get("sum_selections", [])

            self.staging_history = []
            for start, end in self.staged_doc_selections:
                self.staging_history.append(("doc", start, end))
            for start, end in self.staged_sum_selections:
                self.staging_history.append(("sum", start, end))

            self.update_staged_display("doc")
            self.update_staged_display("sum")

            self.remove_annotation_tags(last_ann)

            self.update_annotation_list()
            self.auto_save()
            self.status_label.config(
                text="Undid last annotation and restored to staging."
            )

        else:
            self.status_label.config(text="Nothing to undo.")

    def clear_all_staged(self):
        self.staged_doc_selections, self.staged_sum_selections, self.staging_history = (
            [],
            [],
            [],
        )
        self.update_staged_display("doc")
        self.update_staged_display("sum")
        self.status_label.config(text="Staging area cleared.")

    def confirm_correlation(self):
        if self.view_only_var.get():
            return
        current_doc_idx = self.get_current_doc_index()
        current_doc_widget = self.get_current_doc_widget()

        if not self.staged_doc_selections or not self.staged_sum_selections:
            messagebox.showwarning(
                "Warning", "Stage selections on BOTH sides before confirming."
            )
            return
        if current_doc_idx == -1 or not current_doc_widget:
            messagebox.showwarning("Warning", "No document is selected.")
            return

        correlation_id, color, tag_name = (
            str(uuid.uuid4()),
            self.get_next_color(),
            f"corr-{str(uuid.uuid4())}",
        )
        self.annotations.append(
            {
                "id": correlation_id,
                "color": color,
                "tag_name": tag_name,
                "doc_selections": self.staged_doc_selections,
                "sum_selections": self.staged_sum_selections,
                "document_idx": current_doc_idx,
                "created_at": len(self.annotations),
                "score": 1.0,  # Manual annotations are 100% confidence
                "doc_chunk_scores": [1.0] * len(self.staged_doc_selections),
            }
        )

        current_doc_widget.config(state="normal")
        self.sum_text.config(state="normal")

        # Visual update handled by apply_visual_styles eventually, but set tags initially
        # We rely on apply_visual_styles to set the actual look, but we need the tag present first
        current_doc_widget.tag_config(tag_name, background=color)
        self.sum_text.tag_config(tag_name, background=color)
        for s, event in self.staged_doc_selections:
            current_doc_widget.tag_add(tag_name, s, event)
        for s, event in self.staged_sum_selections:
            self.sum_text.tag_add(tag_name, s, event)

        current_doc_widget.config(state="disabled")
        self.sum_text.config(state="disabled")

        self.clear_all_staged()
        self.update_annotation_list()
        self.auto_save()

        # Ensure new tag is styled correctly
        self.selected_annotation_id = correlation_id
        self.apply_visual_styles()

    def delete_annotation_by_id(self, ann_id):
        if self.view_only_var.get():
            return

        ann = next((a for a in self.annotations if a["id"] == ann_id), None)
        if ann:
            self.remove_annotation_tags(ann)

            self.annotations.remove(ann)
            if self.selected_annotation_id == ann_id:
                self.selected_annotation_id = None

            self.update_annotation_list()
            self.apply_visual_styles()
            self.auto_save()

    def scroll_to_annotation(self, ann_id):
        self.selected_annotation_id = ann_id
        self.apply_visual_styles()

        # Perform Scrolling
        ann = next((a for a in self.annotations if a["id"] == ann_id), None)
        if ann:
            doc_idx = ann.get("document_idx", 0)

            # Find first VISIBLE chunk
            target_pos = None
            doc_selections = ann["doc_selections"]
            chunk_scores = ann.get("doc_chunk_scores", [1.0] * len(doc_selections))
            threshold = self.display_threshold

            for sel, score in zip(doc_selections, chunk_scores):
                if score >= threshold:
                    target_pos = sel[0]
                    break

            # Scroll Document
            if target_pos and 0 <= doc_idx < len(self.doc_text_widgets):
                self.doc_notebook.select(doc_idx)  # Switch to the correct tab
                self.doc_text_widgets[doc_idx].see(target_pos)

            # Scroll Summary
            if ann["sum_selections"]:
                self.sum_text.see(ann["sum_selections"][0][0])

            # Update Status
            score_txt = (
                f" ({int(ann.get('score', 1.0) * 100)}%)" if "score" in ann else ""
            )
            self.status_label.config(
                text=f"Selected correlation {ann_id[:4]}{score_txt}"
            )

    def open_search_dialog(self, _, text: str):
        if self.search_dialog:
            self.search_dialog.focus()
            return
        focused_widget = self.focus_get()
        if isinstance(focused_widget, tk.Text):
            self.search_dialog = SearchDialog(self, focused_widget, text)

    def save_project_as(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Project", "*.json")],
            title="Save Project As...",
        )
        if filepath:
            self.current_project_path = filepath
            self.save_project()

    def save_project(self, is_auto=False):
        if not self.current_project_path:
            return self.save_project_as()

        documents_content = []
        for widget in self.doc_text_widgets:
            widget.config(state="normal")
            documents_content.append(widget.get("1.0", "end-1c"))
            widget.config(state="disabled")

        self.sum_text.config(state="normal")
        data = {
            "documents": documents_content,
            "summary": self.sum_text.get("1.0", "end-1c"),
            "annotations": self.annotations,
        }
        self.sum_text.config(state="disabled")

        try:
            with open(self.current_project_path, "w") as f:
                json.dump(data, f, indent=4)
            message = "Project auto-saved" if is_auto else "Project saved"
            self.status_label.config(text=f"{message} at {time.strftime('%H:%M:%S')}")
            self.file_menu.entryconfig("Save", state="normal")
        except Exception as exc:
            self.status_label.config(text=f"Error saving project: {exc}")

    def on_save(self):
        if not self.current_project_path:
            self.save_project_as()
        self.save_project()

    def auto_save(self):
        if self.autosave_enabled.get() and self.current_project_path:
            self.save_project(is_auto=True)

    def new_project_from_txt(self):
        if self.annotations and not messagebox.askokcancel(
            "New Project", "This will clear your current work. Continue?"
        ):
            return
        doc_path = filedialog.askopenfilename(
            title="Select Source Document (.txt or .json list)",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")],
        )
        if not doc_path:
            return
        sum_path = filedialog.askopenfilename(
            title="Select Summary Document (.txt)",
            filetypes=[("Markdown files", "*.md"), ("Text files", "*.txt")],
        )
        if not sum_path:
            return
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                doc_content = f.read()
                if doc_path.endswith(".json"):
                    docs = json.loads(doc_content)  # Expects a list of strings
                    if not isinstance(docs, list):
                        docs = [str(docs)]  # Handle non-list JSON as a single doc
                else:
                    docs = [doc_content]  # Treat .txt as a list with one doc

            with open(sum_path, "r", encoding="utf-8") as f:
                sum_content = f.read()
            self.reset_all()  # Resets widgets, annotations, etc.

            self.sum_text.config(state="normal")

            # Loop through documents and create a tab for each
            for i, doc_text in enumerate(docs):
                doc_widget, tab_frame = self.create_doc_text_widget(self.doc_notebook)
                doc_widget.config(state="normal")
                doc_widget.insert("1.0", doc_text)
                doc_widget.config(state="disabled")
                tab_title = self._get_doc_title(doc_text, i + 1)
                self.doc_notebook.add(tab_frame, text=tab_title)
                self.doc_text_widgets.append(doc_widget)

            self.sum_text.insert("1.0", sum_content)
            self.sum_text.config(state="disabled")

            self.apply_visual_styles()

            if self.view_only_var.get():
                self.view_only_var.set(False)
                self.toggle_view_only()

            self.status_label.config(
                text="New project loaded. Use File > Save As... to start saving."
            )
            self.show_scores = False
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read text files: {exc}")

    @staticmethod
    def _convert_to_line_char(flat_index: int, lines: list) -> str:
        """Helper to convert flat string index to line.char format"""
        current_offset = 0
        for line_idx, line in enumerate(lines):
            line_len = len(line)
            if flat_index <= current_offset + line_len:
                line_number = line_idx + 1
                char_idx = flat_index - current_offset
                return f"{line_number}.{char_idx}"
            current_offset += line_len
        # Fallback for EOF
        return f"{len(lines)}.0"

    def _load_sim_matrix(self, orig_doc_path: str, doc_chunks_path: str, orig_sum_path: str, sum_chunks_path: str, sim_matrix_path: str, threshold: float):
        try:
            # Load Data
            with open(orig_doc_path, "r", encoding="utf-8") as f:
                original_documents = json.load(f)
            with open(doc_chunks_path, "r", encoding="utf-8") as f:
                doc_chunks = json.load(f)
            with open(orig_sum_path, "r", encoding="utf-8") as f:
                original_summary = f.read()
            with open(sum_chunks_path, "r", encoding="utf-8") as f:
                summary_chunks = json.load(f)
            sim_matrix = np.load(sim_matrix_path)

            # Pre-calculate line splits for index conversion
            doc_lines = [d.splitlines(True) for d in original_documents]
            sum_lines = original_summary.splitlines(True)

            if sim_matrix.shape != (len(doc_chunks), len(summary_chunks)):
                raise ValueError(
                    f"Matrix shape {sim_matrix.shape} mismatch with chunks ({len(doc_chunks)}, {len(summary_chunks)})"
                )

            # Process Matrix
            doc_indices, sum_indices = np.where(sim_matrix > threshold)

            # Group by summary chunk to combine document fragments into single annotations per summary statement
            annotations_dict = {}
            for i, j in zip(doc_indices, sum_indices):
                doc_c = doc_chunks[i]
                sum_c = summary_chunks[j]
                current_score = sim_matrix[i, j]

                # Key: Summary ID (range) + Document ID (to separate correlations across different docs)
                key = f"{sum_c['idx_start']}-{sum_c['idx_end']}.{doc_c['document_idx']}"

                if key not in annotations_dict:
                    annotations_dict[key] = {
                        "sum_chunk": sum_c,
                        "doc_chunks": [],
                        "doc_index": doc_c["document_idx"],
                        "scores": [],
                    }
                annotations_dict[key]["doc_chunks"].append(doc_c)
                annotations_dict[key]["scores"].append(float(current_score))

            # Build Annotations
            new_annotations = []
            for _, val in annotations_dict.items():
                s_chunk = val["sum_chunk"]
                d_chunks = val["doc_chunks"]
                doc_idx = val["doc_index"]
                scores_list = val["scores"]

                # Calculate aggregated score (Max confidence in the group)
                final_score = max(scores_list) if scores_list else 0.0

                # Convert summary indices
                sum_start = self._convert_to_line_char(s_chunk["idx_start"], sum_lines)
                sum_end = self._convert_to_line_char(s_chunk["idx_end"], sum_lines)

                # Convert all doc indices
                doc_selections = []
                for dc in d_chunks:
                    start = self._convert_to_line_char(
                        dc["idx_start"], doc_lines[doc_idx]
                    )
                    end = self._convert_to_line_char(dc["idx_end"], doc_lines[doc_idx])
                    doc_selections.append([start, end])

                correlation_id = str(uuid.uuid4())
                color = self.get_next_color()

                new_annotations.append(
                    {
                        "id": correlation_id,
                        "color": color,
                        "tag_name": f"corr-{correlation_id}",
                        "doc_selections": doc_selections,
                        "doc_chunk_scores": scores_list,
                        "sum_selections": [[sum_start, sum_end]],
                        "document_idx": doc_idx,
                        "created_at": len(new_annotations),
                        "score": final_score,
                    }
                )

            # Load into GUI
            self.reset_all()

            # Load Docs
            self.load_documents(original_documents)

            # Load Summary
            self.sum_text.config(state="normal")
            self.sum_text.insert("1.0", original_summary)
            self.sum_text.config(state="disabled")

            # Apply Annotations
            self.annotations = new_annotations
            for ann in self.annotations:
                tag = ann["tag_name"]
                col = ann["color"]
                d_idx = ann["document_idx"]

                # Apply to Summary
                self.sum_text.tag_config(tag, background=col)
                for s, event in ann["sum_selections"]:
                    self.sum_text.tag_add(tag, s, event)

                # Apply to Document
                if 0 <= d_idx < len(self.doc_text_widgets):
                    dw = self.doc_text_widgets[d_idx]
                    dw.tag_config(tag, background=col)
                    for s, event in ann["doc_selections"]:
                        dw.tag_add(tag, s, event)

            if not self.view_only_var.get():
                self.view_only_var.set(True)
                self.toggle_view_only()

            self.show_scores = True
            self.update_annotation_list()
            self.apply_visual_styles()
            self.status_label.config(
                text=f"Imported {len(new_annotations)} correlations. Please Save As..."
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            messagebox.showerror("Import Error", f"Failed to import from matrix: {exc}")
            self.reset_all()

    def new_project_from_sim_matrix(self):
        if self.annotations and not messagebox.askokcancel(
            "New Project", "This will clear your current work. Continue?"
        ):
            return

        # Ask for all files sequentially
        orig_doc_path = filedialog.askopenfilename(
            title="Select 1/5: Original Documents (JSON)",
            filetypes=[("JSON files", "*.json")],
        )
        if not orig_doc_path:
            return

        doc_chunks_path = filedialog.askopenfilename(
            title="Select 2/5: Document Chunks (JSON)",
            filetypes=[("JSON files", "*.json")],
        )
        if not doc_chunks_path:
            return

        orig_sum_path = filedialog.askopenfilename(
            title="Select 3/5: Original Summary (TXT/MD)",
            filetypes=[
                ("Markdown files", "*.md"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not orig_sum_path:
            return

        sum_chunks_path = filedialog.askopenfilename(
            title="Select 4/5: Summary Chunks (JSON)",
            filetypes=[("JSON files", "*.json")],
        )
        if not sum_chunks_path:
            return

        sim_matrix_path = filedialog.askopenfilename(
            title="Select 5/5: Similarity Matrix (.npy)",
            filetypes=[("Numpy files", "*.npy")],
        )
        if not sim_matrix_path:
            return

        # Ask for threshold
        threshold = simpledialog.askfloat(
            "Threshold",
            "Enter Similarity Threshold (0.0 - 1.0):",
            minvalue=0.0,
            maxvalue=1.0,
            initialvalue=0.5,
        )
        if threshold is None:
            return

        self._load_sim_matrix(orig_doc_path, doc_chunks_path, orig_sum_path, sum_chunks_path, sim_matrix_path, threshold)


    def load_project(self):
        if self.annotations and not messagebox.askokcancel(
            "Load Project", "This will clear your current work. Continue?"
        ):
            return
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON Project", "*.json")], title="Load Annotation Project"
        )
        if not filepath:
            return

        with open(filepath, "r") as f:
            data = json.load(f)
        if not (
            ("document" in data or "documents" in data)
            and "summary" in data
            and "annotations" in data
        ):
            messagebox.showerror("Error", "Invalid project file structure.")
            return
        self.reset_all()

        if "documents" in data:
            docs = data["documents"]
        else:
            docs = [data["document"]]

        self.sum_text.config(state="normal")
        self.load_documents(docs)
        self.sum_text.insert("1.0", data["summary"])
        self.annotations = data["annotations"]
        self.next_color_index = len(self.annotations)

        # Pre-process annotations to ensure backward compatibility and setup IDs
        for ann in self.annotations:
            # Ensure tag name structure
            if "id" not in ann:
                ann["id"] = str(uuid.uuid4())
            tag_name = f"corr-{ann['id']}"
            ann["tag_name"] = tag_name

            # Ensure migration of scores if missing
            if "doc_chunk_scores" not in ann:
                ann["doc_chunk_scores"] = [1.0] * len(ann["doc_selections"])
            if "score" not in ann:
                ann["score"] = 1.0

        for widget in self.doc_text_widgets:
            widget.config(state="disabled")
        self.sum_text.config(state="disabled")

        self.update_annotation_list()
        self.selected_annotation_id = None
        self.apply_visual_styles()
        self.show_scores = False

        if not self.view_only_var.get():
            self.view_only_var.set(True)
            self.toggle_view_only()

        self.current_project_path = filepath
        self.status_label.config(text=f"Project loaded. Saving at: {filepath}")
        self.file_menu.entryconfig("Save", state="normal")

    def reset_all(self):
        for tab_id in self.doc_notebook.tabs():
            self.doc_notebook.forget(tab_id)
        self.doc_text_widgets.clear()

        self.sum_text.config(state="normal")
        self.sum_text.delete("1.0", "end")
        self.sum_text.config(state="disabled")

        # Doc widget tags are cleared when widgets are destroyed
        for tag in self.sum_text.tag_names():
            if tag.startswith("corr-"):
                self.sum_text.tag_delete(tag)

        self.annotations, self.next_color_index = [], 0
        self.selected_annotation_id = None
        self.clear_all_staged()
        self.update_annotation_list()
        self.current_project_path = None
        self.status_label.config(
            text="Welcome! Load a project or create a new one to begin."
        )
        try:
            self.file_menu.entryconfig("Save", state="disabled")
        except (KeyError, AttributeError, tk.TclError):
            pass

        self.show_scores = False

    @staticmethod
    def _get_doc_title(text, index):
        """
        Extracts title from the first line if it's a Markdown H1 (# Title).
        Otherwise, returns generic 'Doc {index}'.
        """
        if not text:
            return f"Doc {index}"

        first_line = text.split("\n", 1)[0].strip()

        # Check for Markdown Level 1 Header
        if first_line.startswith("# "):
            title = first_line[2:].strip()
            # Optional: Truncate if title is extremely long to keep tabs readable
            if len(title) > 15:
                title = title[:12] + "..."
            return f"Doc {index}: {title}"

        return f"Doc {index}"

    def get_next_color(self):
        color = COLORS[self.next_color_index % len(COLORS)]
        self.next_color_index += 1
        return color

    def toggle_view_only(self):
        """Toggles the visibility of the staging/editing area and updates functionality."""
        if self.view_only_var.get():
            # Enable View Only: Hide bottom frame
            self.selection_frame.grid_remove()
            self.clear_all_staged()
            self.status_label.config(text="View Only Mode Enabled.")
        else:
            # Disable View Only: Restore bottom frame
            self.selection_frame.grid()
            self.status_label.config(text="View Only Mode Disabled.")

        # Refresh the sidebar to Remove/Add the "X" (delete) buttons
        self.update_annotation_list()

    def load_documents(self, documents):
        for i, doc_content in enumerate(documents):
            doc_widget, tab_frame = self.create_doc_text_widget(self.doc_notebook)
            doc_widget.config(state="normal")
            doc_widget.insert("1.0", doc_content)
            tab_title = self._get_doc_title(doc_content, i + 1)
            self.doc_notebook.add(tab_frame, text=tab_title)
            self.doc_text_widgets.append(doc_widget)
            doc_widget.config(state="disabled")

    def remove_annotation_tags(self, ann):
        doc_idx = ann.get("document_idx", 0)
        if 0 <= doc_idx < len(self.doc_text_widgets):
            doc_widget = self.doc_text_widgets[doc_idx]
            doc_widget.config(state="normal")
            doc_widget.tag_delete(ann["tag_name"])
            for i in range(len(ann["doc_selections"])):
                doc_widget.tag_delete(f"{ann['tag_name']}_c{i}")
            doc_widget.config(state="disabled")

        self.sum_text.config(state="normal")
        self.sum_text.tag_delete(ann["tag_name"])
        self.sum_text.config(state="disabled")


if __name__ == "__main__":
    app = AnnotationTool()
    app.mainloop()