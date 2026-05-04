# main.py
import os
import re

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

import threading
import queue
import customtkinter as ctk  # type: ignore
import traceback
import json
import datetime
from tkinter import filedialog, messagebox

from ui.ui_components import UISetup, TestDialogs
from core.video_processor import VideoProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DualSubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SimpleAutoSubs — Multimodal Onomatopoeia Subtitler")
        self.message_queue = queue.Queue()
        self.processing_active = False
        self.current_process_index = -1
        self.input_files = []
        self.output_files = []
        self.file_indices = []

        # Logging variables
        self.session_log_path = None

        # Setup crash handling for GUI main loop
        self.root.report_callback_exception = self.handle_gui_exception

        self.setup_ui()
        self.root.after(100, self.process_log_messages)
        self.load_window_geometry()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def handle_gui_exception(self, exc_type, exc_value, exc_traceback):
        error_msg = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback))
        full_msg = f"\n❌ CRITICAL APP CRASH (GUI):\n{error_msg}\n"
        print(full_msg)
        self.log(full_msg)

    def save_window_geometry(self):
        try:
            geometry = self.root.geometry()
            with open("window_geometry.json", "w") as f:
                json.dump({"geometry": geometry}, f)
        except Exception as e:
            self.log(f"Error saving window geometry: {e}")

    def load_window_geometry(self):
        try:
            if os.path.exists("window_geometry.json"):
                with open("window_geometry.json", "r") as f:
                    config = json.load(f)
                    geometry = config.get("geometry")
                    if geometry:
                        self.root.geometry(geometry)
        except Exception as e:
            self.log(f"Error loading window geometry: {e}. Using default.")
            self.root.geometry("700x750")

    def on_closing(self):
        self.save_window_geometry()
        self.root.destroy()

    def setup_ui(self):
        frame = UISetup.create_main_layout(self.root)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(5, weight=1)

        files_frame, self.files_textbox = UISetup.create_file_list_section(
            frame, self)
        files_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        output_frame, self.output_dir_entry = (
            UISetup.create_output_directory_section(frame, self))
        output_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        (
            onomatopoeia_frame,
            self.animation_var,
            self.sync_slider,
        ) = UISetup.create_onomatopoeia_section(frame, self)
        onomatopoeia_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        (
            progress_frame,
            self.progress_label,
            self.progress_bar,
        ) = UISetup.create_progress_section(frame)
        progress_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.process_button = UISetup.create_process_button(frame, self)
        self.process_button.grid(row=4, column=0, pady=10)

        log_frame, self.log_box = UISetup.create_log_section(frame)
        log_frame.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)

    def log(self, message):
        self.message_queue.put(message + "\n")
        if self.session_log_path:
            try:
                with open(self.session_log_path, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception as e:
                print(f"Failed to write to log file: {e}")

    def process_log_messages(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.log_box.insert(ctk.END, message)
                self.log_box.see(ctk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_messages)

    def check_system_status(self):
        TestDialogs.check_system_status(self)

    def add_files(self):
        if len(self.input_files) >= 30:
            messagebox.showwarning(
                "Maximum Files Reached",
                "You can only process up to 30 files at once.",
            )
            return
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Video files", "*.mp4 *.mkv *.avi")])
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path in self.input_files:
                continue
            self.input_files.append(file_path)
            input_basename = os.path.basename(file_path)
            input_name, _ = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            output_dir = self.output_dir_entry.get() or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "output")
            if not self.output_dir_entry.get():
                self.output_dir_entry.insert(0, output_dir)

            output_path = os.path.join(output_dir, output_filename)
            unique_output_path = self.get_unique_output_path(output_path)
            self.output_files.append(unique_output_path)
        self._refresh_files_display()

    def remove_selected_file(self):
        if not self.input_files:
            messagebox.showinfo(
                "No Files", "There are no files to remove.")
            return
        self.input_files.pop()
        self.output_files.pop()
        self._refresh_files_display()

    def clear_all_files(self):
        self.input_files = []
        self.output_files = []
        self._refresh_files_display()

    def _refresh_files_display(self, completed_index=-1):
        """Refreshes the file list display."""
        self.files_textbox.delete("1.0", ctk.END)
        for i, (in_file, out_file) in enumerate(
            zip(self.input_files, self.output_files)
        ):
            display_text = (
                f"{os.path.basename(in_file)} → {os.path.basename(out_file)}"
            )
            if completed_index != -1 and i <= completed_index:
                display_text += " ✓"
            display_text += "\n"
            self.files_textbox.insert(ctk.END, display_text)

    def browse_output_dir(self):
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.output_dir_entry.delete(0, ctk.END)
            self.output_dir_entry.insert(0, output_dir)
            self.update_output_paths(output_dir)

    def update_output_paths(self, output_dir):
        if not self.input_files:
            return
        self.output_files = []
        for input_path in self.input_files:
            input_basename = os.path.basename(input_path)
            input_name, _ = os.path.splitext(input_basename)
            output_filename = f"{input_name}-as.mp4"
            output_path = os.path.join(output_dir, output_filename)
            self.output_files.append(
                self.get_unique_output_path(output_path))
        self._refresh_files_display()

    def get_unique_output_path(self, base_path):
        if not os.path.exists(base_path):
            return base_path
        filename, ext = os.path.splitext(base_path)
        counter = 1
        while os.path.exists(f"{filename}-{counter}{ext}"):
            counter += 1
        return f"{filename}-{counter}{ext}"

    def start_batch_processing_thread(self):
        if not self.input_files:
            messagebox.showinfo(
                "No Files", "Please add at least one video file to process.")
            return
        if self.processing_active:
            messagebox.showinfo(
                "Processing Active",
                "Already processing videos. Please wait.",
            )
            return
        output_dir = self.output_dir_entry.get()
        if not output_dir:
            messagebox.showinfo(
                "No Output Directory",
                "Please select an output directory.",
            )
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Set up per-session log file
        try:
            timestamp = datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M")
            self.session_log_path = os.path.join(
                output_dir, f"{timestamp}.txt")

            with open(self.session_log_path, "w", encoding="utf-8") as f:
                f.write(
                    f"--- BATCH PROCESSING STARTED: {timestamp} ---\n")
                f.write(f"Target Directory: {output_dir}\n")
                f.write(f"Files Queued: {len(self.input_files)}\n")
                f.write("Files to be processed:\n")
                for i, (in_file, out_file) in enumerate(
                    zip(self.input_files, self.output_files)
                ):
                    f.write(
                        f"  {i+1}. {os.path.basename(in_file)} -> "
                        f"{os.path.basename(out_file)}\n"
                    )
                f.write("=" * 60 + "\n\n")

            self.log(
                f"📄 Detailed log will be saved to: "
                f"{self.session_log_path}"
            )
        except Exception as e:
            self.log(f"⚠️ Warning: Could not create log file: {e}")
            self.session_log_path = None

        self.process_button.configure(state="disabled", text="Processing...")
        self.processing_active = True
        threading.Thread(
            target=self.process_all_videos, daemon=True).start()

    def process_all_videos(self):
        try:
            total_videos = len(self.input_files)
            self.log(
                f"Starting batch processing of {total_videos} videos...")
            animation_type = self.animation_var.get()
            sync_offset = self.sync_slider.get()
            detailed_logs = True

            batch_metadata_list = []

            for i, (input_file, output_file) in enumerate(
                zip(self.input_files, self.output_files)
            ):
                self.current_process_index = i

                def update_ui_for_current_file(i=i, input_file=input_file):
                    self.progress_label.configure(
                        text=(
                            f"Processing {i+1}/{total_videos}: "
                            f"{os.path.basename(input_file)}"
                        )
                    )
                    self.progress_bar.set(i / total_videos)
                self.root.after(0, update_ui_for_current_file)

                self.log(
                    f"\n{'='*40}\n"
                    f"PROCESSING VIDEO {i+1}/{total_videos}\n"
                    f"{'='*40}"
                )

                try:
                    env_flag = os.environ.get(
                        "SHORTS_STRATEGIST_ITERATION_LOOP", ""
                    ).strip().lower()
                    iteration_loop_enabled = (
                        env_flag in ("1", "true", "yes", "on")
                        or (
                            getattr(self, "iteration_loop_var", None)
                            and self.iteration_loop_var.get()
                        )
                    )
                    self.log(
                        f"   iteration loop: "
                        f"{'ENABLED' if iteration_loop_enabled else 'disabled'} "
                        f"(env SHORTS_STRATEGIST_ITERATION_LOOP={env_flag!r})"
                    )
                    if iteration_loop_enabled:
                        # Strategist-driven iteration loop. Each video gets
                        # its own metadata file (overwritten across
                        # iterations) so the strategist's edit-review task
                        # can score each one independently.
                        from core.iteration_loop import IterationOrchestrator
                        project_root = os.path.dirname(
                            os.path.abspath(__file__))
                        shorts_data_dir = os.path.join(
                            project_root, "shorts_data")
                        os.makedirs(shorts_data_dir, exist_ok=True)
                        # Pick the next available index for this video's
                        # metadata file. Same logic as the batch path below.
                        max_index = 0
                        for fname in os.listdir(shorts_data_dir):
                            m = re.match(r"shorts_metadata_(\d+)\.json", fname)
                            if m:
                                max_index = max(max_index, int(m.group(1)))
                        per_video_index = max_index + 1 + i
                        per_video_metadata_path = os.path.join(
                            shorts_data_dir,
                            f"shorts_metadata_{per_video_index}.json",
                        )
                        # Per-iteration output filename template.
                        out_dir, out_name = os.path.split(output_file)
                        out_stem, out_ext = os.path.splitext(out_name)
                        output_template = os.path.join(
                            out_dir, f"{out_stem}-v{{iter}}{out_ext}")

                        orch = IterationOrchestrator(
                            max_iterations=3,
                            log_func=self.log,
                        )
                        final_path, single_metadata = orch.run(
                            input_file=input_file,
                            output_file_template=output_template,
                            metadata_path=per_video_metadata_path,
                            animation_type=animation_type,
                            sync_offset=sync_offset,
                            detailed_logs=detailed_logs,
                            final_output_path=output_file,
                        )
                        # Iteration-loop path writes its own per-video metadata
                        # file; do not double-write via the batch list.
                    else:
                        final_path, _, single_metadata = (
                            VideoProcessor.process_single_video(
                                input_file=input_file,
                                output_file=output_file,
                                animation_type=animation_type,
                                sync_offset=sync_offset,
                                detailed_logs=detailed_logs,
                                log_func=self.log,
                            )
                        )
                        if single_metadata:
                            batch_metadata_list.append(single_metadata)

                    self.output_files[i] = final_path

                    input_name = os.path.basename(input_file)
                    output_name = os.path.basename(final_path)
                    self.log("\n" + "*" * 60)
                    self.log(
                        f"SUCCESS: '{input_name}' -> '{output_name}'")
                    self.log("*" * 60 + "\n")

                except Exception as file_error:
                    self.log(
                        f"❌ ERROR processing file "
                        f"{os.path.basename(input_file)}: {file_error}"
                    )
                    self.log(traceback.format_exc())

                def update_ui_on_completion(i=i):
                    self.progress_bar.set((i + 1) / total_videos)
                    self._refresh_files_display(completed_index=i)
                self.root.after(0, update_ui_on_completion)

            # Save batch metadata at end of batch
            if batch_metadata_list:
                try:
                    self.log("\n--- Saving Batch Metadata ---")
                    project_root = os.path.dirname(
                        os.path.abspath(__file__))
                    shorts_data_dir = os.path.join(
                        project_root, "shorts_data")
                    os.makedirs(shorts_data_dir, exist_ok=True)

                    max_index = 0
                    try:
                        for fname in os.listdir(shorts_data_dir):
                            match = re.match(
                                r"shorts_metadata_(\d+)\.json", fname)
                            if match:
                                idx = int(match.group(1))
                                if idx > max_index:
                                    max_index = idx
                    except Exception:
                        pass

                    next_index = max_index + 1
                    json_filename = f"shorts_metadata_{next_index}.json"
                    json_path = os.path.join(
                        shorts_data_dir, json_filename)

                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            batch_metadata_list, f, indent=2, default=str)

                    self.log(f"✅ Batch metadata saved to: {json_path}")
                except Exception as meta_err:
                    self.log(
                        f"⚠️ Failed to save batch metadata: {meta_err}")

            def finalize_ui():
                self.progress_label.configure(
                    text=f"All {total_videos} videos processed!")
                self.process_button.configure(
                    state="normal", text="Process All Videos")
                messagebox.showinfo(
                    "Processing Complete",
                    f"All {total_videos} videos processed!",
                )
                if self.session_log_path:
                    self.log(f"Log file saved: {self.session_log_path}")

            self.root.after(0, finalize_ui)

        except Exception as e:
            self.log(f"🔥 FATAL BATCH PROCESSING CRASH: {e}")
            self.log(traceback.format_exc())

            def reset_ui_on_error():
                self.progress_label.configure(text="Error! See log.")
                self.process_button.configure(
                    state="normal", text="Process All Videos")
            self.root.after(0, reset_ui_on_error)
        finally:
            self.processing_active = False
            self.current_process_index = -1


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = DualSubtitleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()