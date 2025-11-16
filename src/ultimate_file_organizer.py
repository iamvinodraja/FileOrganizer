"""
ultimate_file_organizer_full.py
The ULTIMATE File Organizer with:
- GUI (Tkinter) with Dark/Light themes
- Drag-and-drop folder select + multi-folder support
- Auto-watcher option
- Progress bar and logs
- Password protection (hashed)
- Dropbox backup (optional)
- AI-based categorizer (optional: scikit-learn)
- Packaging notes included below
Required (basic): pip install watchdog plyer playsound tkinterdnd2 dropbox
Optional for AI: pip install scikit-learn joblib
"""

import os
import shutil
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime
from threading import Thread
from queue import Queue, Empty

# GUI and OS integrations
try:
    from tkinter import *
    from tkinter import ttk, filedialog, messagebox, simpledialog
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception as e:
    raise RuntimeError("Tkinter and tkinterdnd2 required. pip install tkinterdnd2") from e

# File watching
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Notifications and sound
from plyer import notification
from playsound import playsound

# Dropbox backup (optional)
try:
    import dropbox
    DROPBOX_AVAILABLE = True
except Exception:
    DROPBOX_AVAILABLE = False

# AI categorizer (optional)
USE_AI = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    import joblib
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False

# ------------------------------
# CONFIG / DEFAULTS
# ------------------------------
APP_DIR = Path.home() / ".ultimate_file_organizer"
APP_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = APP_DIR / "settings.json"
LOG_FILE = APP_DIR / "organizer_log.txt"
PASSWORD_FILE = APP_DIR / "password.hash"

# Default file categories (can be edited via settings)
DEFAULT_FILE_TYPES = {
    "Images": [".png", ".jpg", ".jpeg", ".gif", ".tiff", ".heic", ".bmp", ".svg"],
    "Videos": [".mp4", ".mkv", ".mov", ".flv", ".avi"],
    "Audio": [".mp3", ".wav", ".aac", ".opus", ".flac"],
    "Documents": [".doc", ".docx", ".txt", ".rtf", ".ppt", ".pptx", ".xls", ".xlsx", ".odt"],
    "PDFs": [".pdf"],
    "Archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
    "Code": [".py", ".js", ".html", ".css", ".c", ".cpp", ".java", ".php", ".ts", ".rb", ".go"],
}

# Load or create settings
DEFAULT_SETTINGS = {
    "file_types": DEFAULT_FILE_TYPES,
    "auto_watch": False,
    "sound": True,
    "sound_file": "",  # path to sound file (optional)
    "dropbox_enabled": False,
    "dropbox_token": "",  # user must paste token here to enable
    "theme": "dark",  # or "light"
    "ai_enabled": False,
    "multi_folder": True,
    "copy_to_size_date": True
}

if SETTINGS_FILE.exists():
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        SETTINGS = json.load(f)
else:
    SETTINGS = DEFAULT_SETTINGS.copy()
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(SETTINGS, f, indent=2)


# ------------------------------
# Utilities: logging, notify, sound
# ------------------------------
def log(msg: str):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")


def notify_user(title: str, message: str):
    try:
        notification.notify(title=title, message=message, timeout=4)
    except Exception:
        pass


def play_sound():
    if not SETTINGS.get("sound", True):
        return
    sfile = SETTINGS.get("sound_file", "")
    if sfile and os.path.isfile(sfile):
        try:
            playsound(sfile, block=False)
        except Exception:
            pass


# ------------------------------
# Password protection (hash)
# ------------------------------
def set_password(plain: str):
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, 200000)
    with open(PASSWORD_FILE, "wb") as f:
        f.write(salt + h)
    log("Password set.")


def check_password(plain: str) -> bool:
    if not PASSWORD_FILE.exists():
        return False
    data = PASSWORD_FILE.read_bytes()
    salt = data[:16]
    h = data[16:]
    test = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, 200000)
    return test == h


# ------------------------------
# AI Categorizer (optional)
# ------------------------------
AI_MODEL_PATH = APP_DIR / "ai_model.joblib"

def train_sample_ai_model():
    """
    Very small sample trainer: trains TF-IDF + KNN on example file names & labels.
    Extend with real dataset for better accuracy. Saves to AI_MODEL_PATH.
    """
    if not AI_AVAILABLE:
        raise RuntimeError("scikit-learn and joblib required for AI features.")
    # Example short training data (file name snippets -> class)
    X = [
        "lecture notes linear algebra", "homework assignment math", "physics lab report",
        "holiday photo", "vacation video", "thesis chapter final", "project code main.py",
        "meeting recording", "music track", "resume final", "lecture slides week1"
    ]
    y = [
        "Documents", "Documents", "Documents",
        "Images", "Videos", "Documents", "Code",
        "Audio", "Audio", "Documents", "Documents"
    ]
    vec = TfidfVectorizer()
    Xv = vec.fit_transform(X)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(Xv, y)
    joblib.dump((vec, clf), AI_MODEL_PATH)
    log("AI model trained & saved.")


def ai_predict_category(filename: str):
    if not AI_MODEL_PATH.exists():
        return None
    try:
        vec, clf = joblib.load(AI_MODEL_PATH)
        x = vec.transform([filename])
        pred = clf.predict(x)[0]
        return pred
    except Exception:
        return None


# ------------------------------
# File organizing engine
# ------------------------------
def create_category_folders(base: str, categories: dict):
    for cat in categories.keys():
        os.makedirs(os.path.join(base, cat), exist_ok=True)
    os.makedirs(os.path.join(base, "Others"), exist_ok=True)
    if SETTINGS.get("copy_to_size_date", True):
        os.makedirs(os.path.join(base, "By Size"), exist_ok=True)
        os.makedirs(os.path.join(base, "By Date"), exist_ok=True)


def get_category_for_extension(ext: str, filename: str, categories: dict):
    ext = ext.lower()
    for cat, exts in categories.items():
        if ext in exts:
            return cat
    # Try AI if enabled
    if SETTINGS.get("ai_enabled", False) and AI_AVAILABLE:
        pred = ai_predict_category(filename)
        if pred:
            return pred
    # Fallback: simple keyword rules
    fn = filename.lower()
    if any(k in fn for k in ("lecture", "notes", "assignment", "homework", "report", "essay", "thesis", "resume", "cv")):
        return "Documents"
    if any(k in fn for k in ("screenshot", "img", "photo", "picture", "selfie")):
        return "Images"
    if any(k in fn for k in ("music", "song", "track", "recording")):
        return "Audio"
    return "Others"


def organize_single_file(path: str, base_folder: str, categories: dict, q: Queue=None):
    """
    Moves file into category folder, optionally copies into By Size and By Date folders.
    If q is provided, puts progress messages there.
    """
    if not os.path.isfile(path):
        return False, "Not a file"

    filename = os.path.basename(path)
    try:
        ext = os.path.splitext(filename)[1]
        category = get_category_for_extension(ext, filename, categories)
        create_category_folders(base_folder, categories)

        dst_cat = os.path.join(base_folder, category)
        dst = os.path.join(dst_cat, filename)

        # If destination exists, add counter suffix
        base_name, base_ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dst):
            dst = os.path.join(dst_cat, f"{base_name}_{counter}{base_ext}")
            counter += 1

        shutil.move(path, dst)

        # copy to size/date for convenience
        if SETTINGS.get("copy_to_size_date", True):
            size = os.path.getsize(dst)
            if size < 1_000_000:
                size_folder = "Small (<1MB)"
            elif size < 50_000_000:
                size_folder = "Medium (1–50MB)"
            else:
                size_folder = "Large (>50MB)"
            size_dst_folder = os.path.join(base_folder, "By Size", size_folder)
            os.makedirs(size_dst_folder, exist_ok=True)
            shutil.copy(dst, size_dst_folder)

            created = datetime.fromtimestamp(os.path.getctime(dst)).strftime("%Y-%m-%d")
            date_dst_folder = os.path.join(base_folder, "By Date", created)
            os.makedirs(date_dst_folder, exist_ok=True)
            shutil.copy(dst, date_dst_folder)

        msg = f"Moved {filename} → {dst_cat}"
        log(msg)
        if q:
            q.put(("success", filename))
        return True, msg
    except Exception as e:
        log(f"Error organizing {filename}: {e}")
        if q:
            q.put(("error", filename, str(e)))
        return False, str(e)


# ------------------------------
# Dropbox backup helper
# ------------------------------
def dropbox_upload(file_path: str, dropbox_token: str):
    if not DROPBOX_AVAILABLE:
        raise RuntimeError("Dropbox SDK not installed. pip install dropbox")
    dbx = dropbox.Dropbox(dropbox_token)
    dest_path = f"/OrganizerBackup/{os.path.basename(file_path)}"
    with open(file_path, "rb") as f:
        dbx.files_upload(f.read(), dest_path, mode=dropbox.files.WriteMode.overwrite)
    log(f"Uploaded to Dropbox: {dest_path}")


# ------------------------------
# Watcher for auto-organize
# ------------------------------
class SimpleWatcher(FileSystemEventHandler):
    def __init__(self, base_folder, categories, q=None):
        self.base_folder = base_folder
        self.categories = categories
        self.q = q

    def on_created(self, event):
        if event.is_directory:
            return
        # small delay to allow OS writes to finish
        time.sleep(1)
        organize_single_file(event.src_path, self.base_folder, self.categories, self.q)
        notify_user("File Organizer", f"New file organized in {os.path.basename(self.base_folder)}")


# ------------------------------
# GUI & Threading
# ------------------------------
class OrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate File Organizer - Full")
        self.root.geometry("760x520")
        self.root.resizable(False, False)

        # theme
        self.theme = SETTINGS.get("theme", "dark")
        self.style = ttk.Style(self.root)
        self.apply_theme(self.theme)

        # UI variables
        self.folders = []  # list of folder paths to organize
        self.q = Queue()
        self.observers = []  # for watchers

        # Top frame (folder list + controls)
        top = Frame(self.root)
        top.pack(fill=X, padx=12, pady=8)

        lbl = Label(top, text="Folders to organize (drag-and-drop or browse):")
        lbl.pack(anchor="w")

        # Listbox for multiple folders
        self.listbox = Listbox(top, height=4, selectmode=EXTENDED, width=85)
        self.listbox.pack(side=LEFT, pady=4)
        self.listbox.drop_target_register(DND_FILES)
        self.listbox.dnd_bind("<<Drop>>", self.on_drop)

        scrollbar = Scrollbar(top, command=self.listbox.yview)
        scrollbar.pack(side=LEFT, fill=Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        ctrl_frame = Frame(top)
        ctrl_frame.pack(side=LEFT, padx=8, fill=Y)

        Button(ctrl_frame, text="Add Folder", command=self.add_folder, width=16).pack(pady=4)
        Button(ctrl_frame, text="Remove Selected", command=self.remove_selected, width=16).pack(pady=4)
        Button(ctrl_frame, text="Clear", command=self.clear_folders, width=16).pack(pady=4)

        # Middle frame: actions & progress
        mid = Frame(self.root)
        mid.pack(fill=X, padx=12, pady=6)

        Button(mid, text="Organize Now", command=self.organize_now, width=18).grid(row=0, column=0, padx=6, pady=6)
        Button(mid, text="Start Auto-Watcher", command=self.start_watchers, width=18).grid(row=0, column=1, padx=6, pady=6)
        Button(mid, text="Stop Auto-Watcher", command=self.stop_watchers, width=18).grid(row=0, column=2, padx=6, pady=6)
        Button(mid, text="Settings", command=self.open_settings, width=12).grid(row=0, column=3, padx=6, pady=6)
        Button(mid, text="Set/Reset Password", command=self.set_password_dialog, width=16).grid(row=0, column=4, padx=6, pady=6)

        # Progress area
        pb_frame = Frame(self.root)
        pb_frame.pack(fill=X, padx=12, pady=8)
        self.progress = ttk.Progressbar(pb_frame, orient=HORIZONTAL, length=680, mode='determinate')
        self.progress.pack(pady=6)
        self.status_label = Label(pb_frame, text="Idle")
        self.status_label.pack(anchor="w")

        # Bottom: log viewer
        bottom = Frame(self.root)
        bottom.pack(fill=BOTH, expand=True, padx=12, pady=8)
        Label(bottom, text="Recent Log:").pack(anchor="w")
        self.log_text = Text(bottom, height=10)
        self.log_text.pack(fill=BOTH, expand=True)
        self.refresh_logs_button = Button(bottom, text="Refresh Log", command=self.load_recent_logs)
        self.refresh_logs_button.pack(pady=6, anchor="e")

        # load last settings
        if SETTINGS.get("multi_folder", True):
            pass
        self.load_recent_logs()

        # start a small worker that reads queue and updates UI
        self.root.after(500, self.process_queue)

    # theme application
    def apply_theme(self, theme_name):
        if theme_name == "dark":
            bg = "#1e1e1e"
            fg = "white"
            self.root.configure(bg=bg)
            self.style.configure("TButton", foreground=fg, background="#2d2d2d")
            self.style.configure("TLabel", foreground=fg, background=bg)
            self.style.configure("TProgressbar", background="#00aaff")
        else:
            bg = "white"
            fg = "black"
            self.root.configure(bg=bg)
            self.style.configure("TButton", foreground=fg, background="#f0f0f0")
            self.style.configure("TLabel", foreground=fg, background=bg)

    # drag drop callback for listbox
    def on_drop(self, event):
        data = event.data
        # data might be space-separated list of {path}
        parts = data.split()
        for p in parts:
            p = p.strip("{}")
            if os.path.isdir(p):
                self.add_folder_to_list(p)

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.add_folder_to_list(folder)

    def add_folder_to_list(self, folder):
        if folder not in self.folders:
            self.folders.append(folder)
            self.listbox.insert(END, folder)
            log(f"Added folder: {folder}")

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        for i in reversed(sel):
            self.folders.pop(i)
            self.listbox.delete(i)

    def clear_folders(self):
        self.folders = []
        self.listbox.delete(0, END)

    def load_recent_logs(self):
        self.log_text.delete(1.0, END)
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()[-200:]
                self.log_text.insert(END, "".join(lines))
        else:
            self.log_text.insert(END, "No logs yet.")

    def set_password_dialog(self):
        current_exists = PASSWORD_FILE.exists()
        if current_exists:
            # verify current
            cur = simpledialog.askstring("Password", "Enter current password (cancel to abort):", show='*')
            if cur is None:
                return
            if not check_password(cur):
                messagebox.showerror("Error", "Wrong password.")
                return
        new = simpledialog.askstring("New Password", "Enter new password:", show='*')
        if new:
            set_password(new)
            messagebox.showinfo("Set", "Password set/updated.")

    def require_password(self):
        if not PASSWORD_FILE.exists():
            return True  # no password set
        pw = simpledialog.askstring("Password", "Enter guardian password:", show='*')
        if pw is None:
            return False
        if check_password(pw):
            return True
        messagebox.showerror("Auth Failed", "Wrong password.")
        return False

    def open_settings(self):
        # small modal to edit settings (theme, dropbox, ai flag)
        s = Toplevel(self.root)
        s.title("Settings")
        s.geometry("480x280")
        Label(s, text="Theme:").pack(anchor="w", padx=8, pady=4)
        theme_var = StringVar(value=SETTINGS.get("theme", "dark"))
        ttk.Radiobutton(s, text="Dark", variable=theme_var, value="dark").pack(anchor="w", padx=12)
        ttk.Radiobutton(s, text="Light", variable=theme_var, value="light").pack(anchor="w", padx=12)

        # Dropbox
        Label(s, text="Dropbox (optional): Paste your access token below to enable backups").pack(anchor="w", padx=8, pady=6)
        db_var = StringVar(value=SETTINGS.get("dropbox_token", ""))
        db_entry = Entry(s, textvariable=db_var, width=60)
        db_entry.pack(padx=8, pady=2)
        db_enabled_var = BooleanVar(value=SETTINGS.get("dropbox_enabled", False))
        ttk.Checkbutton(s, text="Enable Dropbox backup", variable=db_enabled_var).pack(anchor="w", padx=12)

        ai_var = BooleanVar(value=SETTINGS.get("ai_enabled", False))
        ttk.Checkbutton(s, text="Enable AI categorizer (requires scikit-learn)", variable=ai_var).pack(anchor="w", padx=12, pady=6)

        def save_and_close():
            SETTINGS["theme"] = theme_var.get()
            SETTINGS["dropbox_token"] = db_var.get().strip()
            SETTINGS["dropbox_enabled"] = bool(db_enabled_var.get())
            SETTINGS["ai_enabled"] = bool(ai_var.get())
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(SETTINGS, f, indent=2)
            self.apply_theme(SETTINGS["theme"])
            s.destroy()
            messagebox.showinfo("Settings", "Saved. Restart app to fully apply some settings.")

        Button(s, text="Save", command=save_and_close).pack(pady=12)

    def organize_now(self):
        if not self.require_password():
            return
        if not self.folders:
            messagebox.showerror("No folders", "Add at least one folder first.")
            return
        # run in background thread, use queue for progress
        t = Thread(target=self._organize_worker, daemon=True)
        t.start()

    def _organize_worker(self):
        total_files = 0
        file_paths = []
        categories = SETTINGS.get("file_types", DEFAULT_FILE_TYPES)
        # collect files first
        for folder in self.folders:
            for fname in os.listdir(folder):
                full = os.path.join(folder, fname)
                if os.path.isfile(full):
                    file_paths.append((full, folder))
        total_files = len(file_paths)
        if total_files == 0:
            messagebox.showinfo("Nothing to do", "No files found in selected folders.")
            return

        self.progress["value"] = 0
        self.progress["maximum"] = total_files
        self.status_label.config(text=f"Organizing {total_files} files...")

        idx = 0
        for path, base_folder in file_paths:
            idx += 1
            organize_single_file(path, base_folder, categories, self.q)
            # dropbox backup if enabled
            if SETTINGS.get("dropbox_enabled") and SETTINGS.get("dropbox_token"):
                try:
                    if DROPBOX_AVAILABLE:
                        dropbox_upload(os.path.join(base_folder, os.path.basename(path)), SETTINGS.get("dropbox_token"))
                    else:
                        log("Dropbox SDK not installed; skipping upload.")
                except Exception as e:
                    log(f"Dropbox upload failed: {e}")

            self.progress["value"] = idx
            self.status_label.config(text=f"Organized {idx}/{total_files}")
            time.sleep(0.05)  # small visual delay

        play_sound()
        notify_user("File Organizer", "Organize complete.")
        self.status_label.config(text="Idle")
        self.load_recent_logs()

    def start_watchers(self):
        if not self.require_password():
            return
        if not self.folders:
            messagebox.showerror("No folders", "Add at least one folder first.")
            return
        # start observer threads
        for folder in self.folders:
            ev = SimpleWatcher(folder, SETTINGS.get("file_types", DEFAULT_FILE_TYPES), self.q)
            obs = Observer()
            obs.schedule(ev, folder, recursive=False)
            obs.daemon = True
            obs.start()
            self.observers.append(obs)
            log(f"Started watcher on {folder}")
        messagebox.showinfo("Watchers", "Auto-watchers started. They will run while this app is open.")

    def stop_watchers(self):
        for obs in self.observers:
            try:
                obs.stop()
                obs.join(0.5)
            except Exception:
                pass
        self.observers = []
        messagebox.showinfo("Watchers", "Auto-watchers stopped.")

    def process_queue(self):
        # handle messages from worker threads
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "success":
                    self.log_text.insert(END, f"OK: {item[1]}\n")
                elif item[0] == "error":
                    self.log_text.insert(END, f"ERR: {item[1]} -> {item[2]}\n")
                self.log_text.see(END)
        except Empty:
            pass
        finally:
            self.root.after(300, self.process_queue)


# ------------------------------
# Packaging & setup notes (printed at runtime)
# ------------------------------
def print_notes():
    notes = f"""
    Setup Notes:
    1) Install required packages:
       pip install watchdog plyer playsound tkinterdnd2 dropbox

       Optional (for AI):
       pip install scikit-learn joblib

    2) To enable Dropbox backups:
       - Create an Access Token at https://www.dropbox.com/developers/apps (choose 'Scoped' app, then generate token)
       - Paste the token into Settings -> Dropbox token and enable checkbox.

    3) To enable AI categorizer (optional):
       - Install scikit-learn & joblib
       - In terminal: python -c "from ultimate_file_organizer_full import train_sample_ai_model; train_sample_ai_model()"
       - Then enable AI in Settings.

    4) Packaging to EXE (Windows) using PyInstaller:
       pip install pyinstaller
       pyinstaller --onefile --noconsole ultimate_file_organizer_full.py
       The EXE will be in dist/

    5) Packaging to DMG (macOS) using py2app:
       pip install py2app
       Create setup.py using py2app docs, then python3 setup.py py2app
       See py2app docs for details.

    6) Password: Use 'Set/Reset Password' in app. If you forget password, delete the file:
       {PASSWORD_FILE}

    7) Logs and settings stored in:
       {APP_DIR}

    """
    print(notes)
    log("Starting application. See printed setup notes.")


# ------------------------------
# Main
# ------------------------------
def main():
    print_notes()
    root = TkinterDnD.Tk()
    app = OrganizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
