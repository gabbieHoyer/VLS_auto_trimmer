import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class VideoBlurTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Blur Tool")
        self.video_path = None
        self.video_basename = None
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.frame = None
        self.original_frame = None
        self.blur_regions = {}
        self.history = {}
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.blur_kernel_size = 51
        self.show_blur = tk.BooleanVar(value=True)
        self.video_queue = []
        self.current_queue_index = -1
        self.save_dir = None  # Store the default save directory

        # GUI Elements
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Sidebar for video queue
        sidebar_frame = tk.Frame(main_frame, width=200)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        tk.Label(sidebar_frame, text="Video Queue").pack(anchor=tk.W)
        self.queue_listbox = tk.Listbox(sidebar_frame, width=30, height=20)
        self.queue_listbox.pack(fill=tk.Y, expand=True)
        self.queue_listbox.bind('<<ListboxSelect>>', self.load_from_queue)

        # Canvas and controls
        content_frame = tk.Frame(main_frame)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(content_frame, width=640, height=480)
        self.canvas.pack(pady=10)

        self.frame_label = tk.Label(content_frame, text="Frame: 0 / 0")
        self.frame_label.pack()

        self.slider = tk.Scale(content_frame, from_=0, to=0, orient=tk.HORIZONTAL, length=600, command=self.slider_update)
        self.slider.pack(pady=5)

        # Blur level slider
        blur_frame = tk.Frame(content_frame)
        blur_frame.pack(pady=5)
        tk.Label(blur_frame, text="Blur Level:").pack(side=tk.LEFT)
        self.blur_slider = tk.Scale(blur_frame, from_=11, to=101, orient=tk.HORIZONTAL, length=200, command=self.update_blur_level)
        self.blur_slider.set(self.blur_kernel_size)
        self.blur_slider.pack(side=tk.LEFT, padx=5)
        self.blur_label = tk.Label(blur_frame, text=f"Kernel: {self.blur_kernel_size}x{self.blur_kernel_size}")
        self.blur_label.pack(side=tk.LEFT, padx=5)

        # Blur preview toggle
        self.blur_toggle = tk.Checkbutton(content_frame, text="Show Blur", variable=self.show_blur)
        self.blur_toggle.pack(pady=5)
        self.show_blur.trace_add("write", lambda *args: self.update_frame())

        btn_frame = tk.Frame(content_frame)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Load Video", command=self.confirm_load_new_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Select Save Folder", command=self.select_save_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Previous Frame", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        self.next_video_btn = tk.Button(btn_frame, text="Next Video", command=self.load_next_video, state=tk.DISABLED)
        self.next_video_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Video", command=self.save_video).pack(side=tk.LEFT, padx=5)
        self.undo_btn = tk.Button(btn_frame, text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        self.redo_btn = tk.Button(btn_frame, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_btn.pack(side=tk.LEFT, padx=5)

        # Bind mouse events to canvas
        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)

    def select_save_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.save_dir = folder_path

    def update_blur_level(self, val):
        self.blur_kernel_size = int(val)
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
        self.blur_slider.set(self.blur_kernel_size)
        self.blur_label.configure(text=f"Kernel: {self.blur_kernel_size}x{self.blur_kernel_size}")
        self.update_frame()

    def confirm_load_new_video(self, video_path=None, from_queue=False):
        if self.cap and self.blur_regions:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved blur regions. Do you want to save the current video before loading a new one?",
                default="yes"
            )
            if response is True:
                self.save_video()
                if self.cap:
                    self.load_video(video_path, from_queue)
            elif response is False:
                self.load_video(video_path, from_queue)
        else:
            self.load_video(video_path, from_queue)

    def load_video(self, video_path=None, from_queue=False):
        if not video_path:
            video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mp4")])
        if not video_path:
            return
        if self.cap:
            self.cap.release()
            self.cap = None
        self.blur_regions = {}
        self.history = {}
        self.current_frame = 0
        self.total_frames = 0
        self.frame = None
        self.original_frame = None
        self.blur_kernel_size = 51
        self.blur_slider.set(self.blur_kernel_size)
        self.blur_label.configure(text=f"Kernel: {self.blur_kernel_size}x{self.blur_kernel_size}")
        self.slider.set(0)
        self.slider.configure(to=0)
        self.frame_label.configure(text="Frame: 0 / 0")
        self.canvas.delete("all")
        self.undo_btn.configure(state=tk.DISABLED)
        self.redo_btn.configure(state=tk.DISABLED)
        if not from_queue:
            self.video_queue = []
            self.current_queue_index = -1
            self.queue_listbox.delete(0, tk.END)
            self.save_dir = None  # Reset save directory when loading a new single video
        self.video_path = video_path
        self.video_basename = os.path.splitext(os.path.basename(video_path))[0]
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            self.cap = None
            self.video_path = None
            self.video_basename = None
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.configure(to=self.total_frames - 1)
        self.frame_label.configure(text=f"Frame: 0 / {self.total_frames}")
        if from_queue:
            self.update_queue_selection()
        self.update_frame()

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        self.video_queue = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.mp4', '.avi'))
        ]
        if not self.video_queue:
            messagebox.showwarning("Warning", "No video files found in the selected folder")
            return
        self.queue_listbox.delete(0, tk.END)
        for video_path in self.video_queue:
            self.queue_listbox.insert(tk.END, os.path.basename(video_path))
        self.current_queue_index = 0
        self.save_dir = None  # Reset save directory when loading a new folder
        self.confirm_load_new_video(self.video_queue[0], from_queue=True)
        self.next_video_btn.configure(state=tk.NORMAL)

    def load_from_queue(self, event):
        selection = self.queue_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if index != self.current_queue_index:
            self.current_queue_index = index
            self.confirm_load_new_video(self.video_queue[index], from_queue=True)

    def load_next_video(self):
        if not self.video_queue or self.current_queue_index < 0:
            return
        self.current_queue_index = (self.current_queue_index + 1) % len(self.video_queue)
        self.confirm_load_new_video(self.video_queue[self.current_queue_index], from_queue=True)

    def update_queue_selection(self):
        self.queue_listbox.selection_clear(0, tk.END)
        if self.current_queue_index >= 0:
            self.queue_listbox.selection_set(self.current_queue_index)
            self.queue_listbox.see(self.current_queue_index)

    def update_frame(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.original_frame = frame.copy()
        self.frame = self.original_frame.copy()
        if self.show_blur.get() and self.current_frame in self.blur_regions:
            for (x1, y1, x2, y2) in self.blur_regions[self.current_frame]:
                roi = self.frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(roi, (self.blur_kernel_size, self.blur_kernel_size), 0)
                    self.frame[y1:y2, x1:x2] = blurred_roi
        if self.current_frame in self.blur_regions:
            for (x1, y1, x2, y2) in self.blur_regions[self.current_frame]:
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(frame_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.frame_label.configure(text=f"Frame: {self.current_frame} / {self.total_frames}")
        self.slider.set(self.current_frame)
        self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        undo_stack = self.history.get(self.current_frame, {"undo": [], "redo": []})["undo"]
        redo_stack = self.history.get(self.current_frame, {"undo": [], "redo": []})["redo"]
        self.undo_btn.configure(state=tk.NORMAL if undo_stack else tk.DISABLED)
        self.redo_btn.configure(state=tk.NORMAL if redo_stack else tk.DISABLED)

    def slider_update(self, val):
        self.current_frame = int(val)
        self.update_frame()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.update_frame()

    def start_rect(self, event):
        self.drawing = True
        self.ix = int(event.x * (self.frame.shape[1] / 640))
        self.iy = int(event.y * (self.frame.shape[0] / 480))

    def draw_rect(self, event):
        if self.drawing:
            x = int(event.x * (self.frame.shape[1] / 640))
            y = int(event.y * (self.frame.shape[0] / 480))
            x1, x2 = min(self.ix, x), max(self.ix, x)
            y1, y2 = min(self.iy, y), max(self.iy, y)
            frame_copy = self.original_frame.copy()
            if self.show_blur.get():
                roi = frame_copy[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(roi, (self.blur_kernel_size, self.blur_kernel_size), 0)
                    frame_copy[y1:y2, x1:x2] = blurred_roi
            if self.current_frame in self.blur_regions:
                for (rx1, ry1, rx2, ry2) in self.blur_regions[self.current_frame]:
                    if self.show_blur.get():
                        roi = frame_copy[ry1:ry2, rx1:rx2]
                        if roi.size > 0:
                            blurred_roi = cv2.GaussianBlur(roi, (self.blur_kernel_size, self.blur_kernel_size), 0)
                            frame_copy[ry1:ry2, rx1:rx2] = blurred_roi
                    cv2.rectangle(frame_copy, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(frame_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def end_rect(self, event):
        if self.drawing:
            self.drawing = False
            x = int(event.x * (self.frame.shape[1] / 640))
            y = int(event.y * (self.frame.shape[0] / 480))
            x1, x2 = min(self.ix, x), max(self.ix, x)
            y1, y2 = min(self.iy, y), max(self.iy, y)
            rect = (x1, y1, x2, y2)
            if self.current_frame not in self.blur_regions:
                self.blur_regions[self.current_frame] = []
            self.blur_regions[self.current_frame].append(rect)
            if self.current_frame not in self.history:
                self.history[self.current_frame] = {"undo": [], "redo": []}
            self.history[self.current_frame]["undo"].append(("add", rect))
            self.history[self.current_frame]["redo"] = []
            self.update_frame()

    def undo(self):
        if self.current_frame not in self.history or not self.history[self.current_frame]["undo"]:
            return
        action, rect = self.history[self.current_frame]["undo"].pop()
        if action == "add":
            if self.current_frame in self.blur_regions and rect in self.blur_regions[self.current_frame]:
                self.blur_regions[self.current_frame].remove(rect)
                if not self.blur_regions[self.current_frame]:
                    del self.blur_regions[self.current_frame]
            self.history[self.current_frame]["redo"].append(("remove", rect))
        elif action == "remove":
            if self.current_frame not in self.blur_regions:
                self.blur_regions[self.current_frame] = []
            self.blur_regions[self.current_frame].append(rect)
            self.history[self.current_frame]["redo"].append(("add", rect))
        self.update_frame()

    def redo(self):
        if self.current_frame not in self.history or not self.history[self.current_frame]["redo"]:
            return
        action, rect = self.history[self.current_frame]["redo"].pop()
        if action == "add":
            if self.current_frame not in self.blur_regions:
                self.blur_regions[self.current_frame] = []
            self.blur_regions[self.current_frame].append(rect)
            self.history[self.current_frame]["undo"].append(("add", rect))
        elif action == "remove":
            if self.current_frame in self.blur_regions and rect in self.blur_regions[self.current_frame]:
                self.blur_regions[self.current_frame].remove(rect)
                if not self.blur_regions[self.current_frame]:
                    del self.blur_regions[self.current_frame]
            self.history[self.current_frame]["undo"].append(("remove", rect))
        self.update_frame()

    def save_video(self):
        if not self.cap:
            messagebox.showwarning("Warning", "No video loaded")
            return
        if not self.blur_regions:
            response = messagebox.askyesno(
                "No Blur Regions",
                "No blur objects created. Are you sure you still want to save the video?",
                default="no"
            )
            if not response:
                return
        default_output = f"{self.video_basename}_anon.mp4" if self.video_basename else "output_anon.mp4"
        initial_dir = self.save_dir if self.save_dir else (os.path.dirname(self.video_path) if self.video_path else None)
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            initialfile=default_output,
            initialdir=initial_dir
        )
        if not output_path:
            return
        # Update save_dir to the directory of the selected output path
        self.save_dir = os.path.dirname(output_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx in self.blur_regions:
                for (x1, y1, x2, y2) in self.blur_regions[frame_idx]:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (self.blur_kernel_size, self.blur_kernel_size), 0)
                        frame[y1:y2, x1:x2] = blurred_roi
            out.write(frame)
        out.release()
        messagebox.showinfo("Success", "Video saved successfully")
        self.blur_regions = {}
        self.history = {}
        self.update_frame()

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoBlurTool(root)
    root.mainloop()

# python -m src.app.manual_tools.video_blur_tool