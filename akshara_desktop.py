import os
import threading
import time
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from docx.shared import Pt

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "kannada_telugu_ocr_config.json"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

POPPLER_PATH = r"C:\Users\Suresh Sharma\OneDrive\Desktop\OCR_Tools\poppler-24.02.0\Library\bin"

class MultiScriptOCR(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AksharaDrishti - Kannada, Telugu & Sanskrit OCR")
        self.geometry("1100x800")

        self.cancel_flag = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(self, text="AksharaDrishti", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0, pady=20)

        # Input
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(input_frame, text="Input (PDF or Folder):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.input_entry = ctk.CTkEntry(input_frame)
        self.input_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(input_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=10)

        # Output
        output_frame = ctk.CTkFrame(self)
        output_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        output_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(output_frame, text="Output Folder:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.output_entry = ctk.CTkEntry(output_frame)
        self.output_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=2, padx=10)

        # Options
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(options_frame, text="Language:").grid(row=0, column=0, padx=20, pady=10)
        self.lang_var = ctk.StringVar(value="kan+san")
        ctk.CTkOptionMenu(options_frame, values=["kan+san", "kan", "tel+san", "tel", "san", "kan+eng"], variable=self.lang_var).grid(row=0, column=1, padx=10)

        ctk.CTkLabel(options_frame, text="DPI:").grid(row=0, column=2, padx=20, pady=10)
        self.dpi_var = ctk.IntVar(value=350)
        ctk.CTkEntry(options_frame, textvariable=self.dpi_var, width=100).grid(row=0, column=3, padx=10)

        # Status
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(status_frame)
        self.progress.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.progress.set(0)

        self.time_label = ctk.CTkLabel(status_frame, text="Time: 0s | Estimated: --", font=ctk.CTkFont(size=14))
        self.time_label.grid(row=1, column=0, pady=(0, 10))

        self.current_page_label = ctk.CTkLabel(status_frame, text="Page: 0 / 0", font=ctk.CTkFont(size=14))
        self.current_page_label.grid(row=2, column=0, pady=(0, 10))

        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        self.start_btn = ctk.CTkButton(button_frame, text="Start OCR", fg_color="green", command=self.start_ocr_thread, width=200, height=50)
        self.start_btn.grid(row=0, column=0, padx=30, pady=10)

        self.cancel_btn = ctk.CTkButton(button_frame, text="Cancel", fg_color="red", command=self.cancel_ocr, state="disabled", width=150, height=50)
        self.cancel_btn.grid(row=0, column=1, padx=30, pady=10)

        # Log
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)

        self.log("Ready! Supports Kannada & Telugu + Sanskrit. Progress & time tracking active.")

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.update_idletasks()

    def browse_input(self):
        path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF Files", "*.pdf")]) or filedialog.askdirectory()
        if path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def start_ocr_thread(self):
        if not self.input_entry.get().strip() or not self.output_entry.get().strip():
            messagebox.showerror("Error", "Select input and output!")
            return
        self.cancel_flag = False
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        self.time_label.configure(text="Time: 0s | Estimated: --")
        self.current_page_label.configure(text="Page: 0 / 0")
        threading.Thread(target=self.perform_ocr, daemon=True).start()

    def cancel_ocr(self):
        self.cancel_flag = True
        self.log("Cancelled")

from ocr_utils import preprocess_image, process_image_to_docx_content

    # preprocess_image method removed as we use the imported one

    def perform_ocr(self):
        start_time = time.time()
        try:
            input_path = self.input_entry.get().strip()
            output_dir = self.output_entry.get().strip()
            lang = self.lang_var.get()
            dpi = self.dpi_var.get()
            config = r'--oem 1 -c preserve_interword_spaces=1' # Removed psm 6

            pdf_files = []
            if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
                pdf_files = [input_path]
            elif os.path.isdir(input_path):
                pdf_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".pdf")]

            if not pdf_files:
                self.log("No PDF found!")
                return

            total_processed = 0
            total_pages_all = 0

            for pdf_path in pdf_files:
                images_temp = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
                total_pages_all += len(images_temp)

            for pdf_path in pdf_files:
                if self.cancel_flag:
                    break
                filename = os.path.basename(pdf_path)
                self.log(f"Processing: {filename}")

                images = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH)
                num_pages = len(images)
                doc = Document()

                # Choose font based on language
                if "tel" in lang:
                    font_name = "Noto Sans Telugu"  # Best Telugu font
                else:
                    font_name = "Noto Sans Kannada"

                normal_style = doc.styles['Normal']
                normal_style.font.name = font_name
                normal_style.font.size = Pt(12)

                for i, img in enumerate(images):
                    if self.cancel_flag:
                        break

                    page_start = time.time()
                    processed = preprocess_image(img) # Use imported function
                    
                    # Use shared logic for formatting
                    process_image_to_docx_content(doc, processed, lang, config)
                    
                    page_time = time.time() - page_start
                    
                    # Page break logic handled by check below, but process_image_to_docx_content writes content
                    if i < num_pages - 1:
                        doc.add_page_break()

                    total_processed += 1
                    progress = total_processed / total_pages_all if total_pages_all > 0 else 0
                    self.progress.set(progress)

                    elapsed = int(time.time() - start_time)
                    avg_per_page = elapsed / total_processed if total_processed > 0 else 0
                    remaining = total_pages_all - total_processed
                    est_remaining = int(avg_per_page * remaining)

                    self.current_page_label.configure(text=f"Page: {total_processed} / {total_pages_all}")
                    self.time_label.configure(text=f"Time: {elapsed}s | Est. remaining: {est_remaining}s")

                    self.log(f"Completed page {i+1}/{num_pages} ({page_time:.1f}s)")

                if not self.cancel_flag:
                    suffix = "_TEL" if "tel" in lang else "_KAN"
                    out_file = os.path.join(output_dir, filename.replace(".pdf", f"{suffix}_OCR.docx"))
                    doc.save(out_file)
                    self.log(f"Saved: {os.path.basename(out_file)}")

            self.log("All done!")

        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

if __name__ == "__main__":
    app = MultiScriptOCR()
    app.mainloop()