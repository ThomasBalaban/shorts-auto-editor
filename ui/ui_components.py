# ui/ui_components.py - UPDATED
"""
UI components and setup for the shorts-auto-editor application.
Simplified for automatic dual-track transcription and onomatopoeia.
"""

import customtkinter as ctk # type: ignore
from tkinter import messagebox


class UISetup:
    """Handles UI setup and component creation."""

    @staticmethod
    def create_main_layout(root):
        """Create the main scrollable layout."""
        main_scroll_container = ctk.CTkScrollableFrame(root, width=660, height=600)
        main_scroll_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(main_scroll_container)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        return frame

    @staticmethod
    def create_file_list_section(parent, app):
        """Create the file list section. Returns the containing frame and the textbox widget."""
        files_frame = ctk.CTkFrame(parent)

        ctk.CTkLabel(files_frame, text="Input Files:").pack(anchor="w", pady=5)

        list_frame = ctk.CTkFrame(files_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)

        files_textbox = ctk.CTkTextbox(list_frame, height=120, width=600)
        files_textbox.pack(fill="both", expand=True)

        # File list buttons
        button_frame = ctk.CTkFrame(files_frame)
        button_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(button_frame, text="Add Files", command=app.add_files).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(button_frame, text="Remove Last", command=app.remove_selected_file).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(button_frame, text="Clear All", command=app.clear_all_files).pack(side="left", padx=5, pady=5)

        ctk.CTkButton(
            button_frame,
            text="System Status",
            command=app.check_system_status,
            width=120
        ).pack(side="right", padx=5, pady=5)

        return files_frame, files_textbox

    @staticmethod
    def create_output_directory_section(parent, app):
        """Create the output directory selection section. Returns the containing frame and the entry widget."""
        output_frame = ctk.CTkFrame(parent)

        ctk.CTkLabel(output_frame, text="Output Directory:").pack(side="left", padx=5, pady=5)
        output_dir_entry = ctk.CTkEntry(output_frame, width=400)
        output_dir_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        ctk.CTkButton(output_frame, text="Browse", command=app.browse_output_dir).pack(side="left", padx=5, pady=5)

        return output_frame, output_dir_entry

    @staticmethod
    def create_onomatopoeia_section(parent, app):
        """Create the onomatopoeia settings section. Returns the frame, var, and slider."""
        onomatopoeia_frame = ctk.CTkFrame(parent)
        ctk.CTkLabel(onomatopoeia_frame, text="Onomatopoeia (Comic Book Effects):", font=("Arial", 12, "bold")).pack(anchor="w", padx=5, pady=(5,0))

        # Animation Settings
        animation_row = ctk.CTkFrame(onomatopoeia_frame)
        animation_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(animation_row, text="Animation Type:").pack(side="left", padx=5, pady=5)
        animation_var = ctk.StringVar(value="Auto")

        animation_options = [
            "Drift & Fade", "Wiggle", "Pop & Shrink",
            "Shake", "Pulse", "Wave", "Explode-Out", "Hyper Bounce", "Static"
        ]

        ctk.CTkOptionMenu(
            animation_row,
            values=animation_options,
            variable=animation_var,
            width=150
        ).pack(side="left", padx=5, pady=5)
        
        # Sync Offset Slider
        sync_row = ctk.CTkFrame(onomatopoeia_frame)
        sync_row.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(sync_row, text="Sync Offset (s):").pack(side="left", padx=5)
        
        # Default -0.15 for M4 Mac drift
        sync_slider = ctk.CTkSlider(sync_row, from_=-0.5, to=0.5, number_of_steps=20)
        sync_slider.set(-0.15) 
        sync_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        sync_value_label = ctk.CTkLabel(sync_row, text="-0.15s")
        sync_value_label.pack(side="left", padx=5)
        
        def update_label(value):
            sync_value_label.configure(text=f"{value:.2f}s")
            
        sync_slider.configure(command=update_label)

        return onomatopoeia_frame, animation_var, sync_slider

    @staticmethod
    def create_progress_section(parent):
        """Create the progress indicator section. Returns the containing frame and widgets."""
        progress_frame = ctk.CTkFrame(parent)

        progress_label = ctk.CTkLabel(progress_frame, text="Ready")
        progress_label.pack(side="top", pady=2)

        progress_bar = ctk.CTkProgressBar(progress_frame, width=600)
        progress_bar.pack(side="top", pady=5, fill="x")
        progress_bar.set(0)

        return progress_frame, progress_label, progress_bar

    @staticmethod
    def create_process_button(parent, app):
        """Create the main process button."""
        return ctk.CTkButton(
            parent,
            text="Process All Videos",
            command=app.start_batch_processing_thread,
            height=40,
            font=("Arial", 14, "bold")
        )

    @staticmethod
    def create_log_section(parent):
        """Create the log area. Returns the containing frame and the textbox widget."""
        log_frame = ctk.CTkFrame(parent)
        ctk.CTkLabel(log_frame, text="Processing Log:").pack(anchor="w", pady=(10, 0))
        log_box = ctk.CTkTextbox(log_frame, height=450, width=600)
        log_box.pack(fill="both", expand=True, padx=5, pady=5)
        return log_frame, log_box


class TestDialogs:
    """Simplified system checking."""
    # This class remains unchanged
    @staticmethod
    def check_system_status(app):
        try:
            app.log("="*50)
            app.log("SYSTEM STATUS CHECK")
            app.log("="*50)
            from core.transcriber import WhisperModel
            app.log("✅ Whisper (for dialogue): OPERATIONAL")
            from onomatopoeia_detector import OnomatopoeiaDetector
            detector = OnomatopoeiaDetector(log_func=app.log)
            app.log("✅ Onomatopoeia System: OPERATIONAL")
            import torch # type: ignore
            if torch.backends.mps.is_available():
                app.log("✅ GPU (Mac MPS): Available")
            elif torch.cuda.is_available():
                app.log("✅ GPU (Nvidia CUDA): Available")
            else:
                app.log("⚠️  GPU: Not available (using CPU, will be slower)")
            messagebox.showinfo(
                "System Status",
                "System check complete. See the log for details.\n\n"
                "Ensure all systems are operational before processing."
            )
        except Exception as e:
            error_msg = f"Error checking system status: {e}"
            app.log(error_msg)
            messagebox.showerror("Check Error", error_msg)