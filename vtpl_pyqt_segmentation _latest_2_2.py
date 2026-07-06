import sys
import os

from PyQt5.QtCore import Qt, QPoint, QLibraryInfo

# Force PyQt5 to use its own Qt plugins
plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(plugin_path, "platforms")
os.environ["QT_PLUGIN_PATH"] = plugin_path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QWidget, QComboBox, QSlider, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage

import cv2
import numpy as np
from datetime import datetime                                          

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setStyleSheet("border:none; background:transparent;")
        self.setFocusPolicy(Qt.StrongFocus)

        self.image = None
        self.mask = None
        self.temp_mask = None

        self.scale = 1.0
        self.zoom_step = 1.15
        self.min_scale = 0.2
        self.max_scale = 2.0

        self.drawing = False
        self.last_point = QPoint()

        self.brush_size = 5
        self.current_class_value = 255
        self.eraser_mode = False

        self.polygon_mode = False
        self.polygon_points = []

        self.mask_update_callback = None
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50

        self.main_window = None
        self.mouse_pos = None
        self.mask_modified = False

    def push_undo(self):
        """Saves current state (mask + polygon points) to undo stack."""
        if self.mask is not None:
            # We store a tuple of (mask_copy, points_copy)
            self.undo_stack.append((self.mask.copy(), list(self.polygon_points)))
            self.redo_stack.clear()
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)

    def undo(self):  # new add polygon undo
        if self.undo_stack:
            self.redo_stack.append((self.mask.copy(), list(self.polygon_points)))
            self.mask, self.polygon_points = self.undo_stack.pop()
            self.temp_mask = self.mask.copy()
            self.mask_modified = True
            self.update_overlay()

    def redo(self):    # new add polygon redo
        if self.redo_stack:
            self.undo_stack.append((self.mask.copy(), list(self.polygon_points)))
            self.mask, self.polygon_points = self.redo_stack.pop()
            self.temp_mask = self.mask.copy()
            self.mask_modified = True
            self.update_overlay()

    def set_images(self, image, mask):
        self.image = image
        self.mask = mask
        self.temp_mask = mask.copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.polygon_points = []
        self.scale = 1.0
        self.mask_modified = False
        self.update_overlay()
        self.setFocus()

    def update_overlay(self):
        if self.image is None or self.mask is None:
            return
        
        # Determine which base mask to show
        use_mask = self.temp_mask if self.polygon_mode else self.mask
        use_mask = self.mask
        color_mask = cv2.applyColorMap(use_mask.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(self.image, 0.7, color_mask, 0.3, 0)

        # Draw Polygon UI =========
        if self.polygon_mode and len(self.polygon_points) > 0:
            pts = np.array(self.polygon_points, np.int32)
            if len(pts) > 1:
                cv2.polylines(overlay, [pts], False, (0, 0, 255), 3)
            for x, y in self.polygon_points:
                cv2.circle(overlay, (x, y), 8, (0, 0, 0), -1)
                cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)

        # Draw Brush Preview =======
        if self.mouse_pos is not None and not self.polygon_mode:
            x, y = self.mouse_pos
            radius = max(1, int(self.brush_size))
            thickness = max(1, int(self.brush_size / self.scale))
            radius = max(1, thickness // 2)
            cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1)

        self.display_image(overlay)
        if self.mask_update_callback:
            self.mask_update_callback(use_mask)

    def display_image(self, img):
        h, w, _ = img.shape
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        qimg = QImage(img_resized.data, new_w, new_h, new_w * 3, QImage.Format_RGB888).rgbSwapped()
        qimg = QImage(img_resized.data, new_w, new_h, img_resized.strides[0], QImage.Format_RGB888).rgbSwapped()
        self.setPixmap(QPixmap.fromImage(qimg))
        self.setFixedSize(new_w, new_h)

    def wheelEvent(self, event):
        if self.image is None: return
        self.scale *= self.zoom_step if event.angleDelta().y() > 0 else 1 / self.zoom_step
        self.scale = max(self.min_scale, min(self.max_scale, self.scale))
        self.update_overlay()

    def _to_img(self, pos):
        return int(pos.x() / self.scale), int(pos.y() / self.scale)

    def mousePressEvent(self, event):
        self.setFocus()
        if self.mask is None: return
        if event.button() == Qt.LeftButton:
            self.push_undo() # Capture state before adding a point or stroke
            if self.polygon_mode:
                x, y = self._to_img(event.pos())
                self.polygon_points.append((x, y))
                self.temp_mask = self.mask.copy()
                self.update_overlay()
            else:
                self.drawing = True
                self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.image is None: return
        self.mouse_pos = self._to_img(event.pos())
        if self.drawing and not self.polygon_mode:
            p1 = self._to_img(self.last_point)
            p2 = self._to_img(event.pos())
            val = 0 if self.eraser_mode else self.current_class_value
            cv2.line(self.mask, p1, p2, val, max(1, int(self.brush_size / self.scale)))
            self.mask_modified = True
            self.last_point = event.pos()
        self.update_overlay()

    def mouseReleaseEvent(self, event):
        self.drawing = False

    def fill_polygon(self):
        if len(self.polygon_points) >= 3:
            # We don't necessarily need another push_undo here because the state
            # was pushed when the points were added, but it's safer to push 
            # the state with the points before they are cleared.
            self.push_undo()
            cv2.fillPoly(self.mask, [np.array(self.polygon_points, np.int32)], self.current_class_value)
            self.mask_modified = True
            self.temp_mask = self.mask.copy()
            self.polygon_points.clear()
            self.update_overlay()

    def keyPressEvent(self, event):       # key event ======
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_S: self.main_window.save_mask(); return
            if event.key() == Qt.Key_Z: self.undo(); return
            if event.key() == Qt.Key_Y: self.redo(); return
        if self.main_window:
            if event.key() == Qt.Key_Up: self.main_window.load_relative(-1)
            elif event.key() == Qt.Key_Down: self.main_window.load_relative(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation Editor")
        self.resize(1600, 900)

        self.btn_style = """
            QPushButton { background-color: none; padding: 5px; border: 1px solid #999; border-radius: 3px; }
            QPushButton:checked { background-color: #3498db; color: white; font-weight: bold; border: 1px solid #2980b9; }
        """

        # UI
        self.image_name_label = QLabel("Image: No file loaded")
        self.image_name_label.setStyleSheet("font-size:16px; font-weight:bold; padding: 5px;")
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search file name...")
        self.search_bar.setStyleSheet("padding: 5px; border: 1px solid #ccc; margin-bottom: 5px;")
        self.search_bar.returnPressed.connect(self.search_image)

        self.input_label = QLabel()
        self.mask_label = QLabel()
        self.editor = ImageLabel()
        self.editor.mask_update_callback = self.update_mask_view
        self.editor.main_window = self

        self.load_btn = QPushButton("Load Image + Mask")
        self.save_btn = QPushButton("Save Mask")
        self.fill_btn = QPushButton("Fill Polygon")
        
        self.poly_btn = QPushButton("Polygon Mode")
        self.poly_btn.setCheckable(True)
        
        self.eraser_btn = QPushButton("Eraser")
        self.eraser_btn.setCheckable(True)

        self.poly_btn.setStyleSheet(self.btn_style)
        self.eraser_btn.setStyleSheet(self.btn_style)

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 100)
        self.brush_slider.setValue(5)
        self.brush_slider.setFixedWidth(150)

        self.class_box = QComboBox()
        self.class_box.addItems(["0", "127", "220"])   # previous 255 

        # Connections
        self.load_btn.clicked.connect(self.load_images)
        self.save_btn.clicked.connect(self.save_mask)
        self.fill_btn.clicked.connect(self.editor.fill_polygon)
        self.poly_btn.toggled.connect(self.toggle_polygon)
        self.eraser_btn.toggled.connect(self.toggle_eraser)
        self.brush_slider.valueChanged.connect(lambda v: setattr(self.editor, "brush_size", v))
        self.class_box.currentTextChanged.connect(lambda v: setattr(self.editor, "current_class_value", int(v)))

        # Layout
        side_panel = QVBoxLayout()
        side_panel.addWidget(QLabel("SEARCH:"))
        side_panel.addWidget(self.search_bar)
        side_panel.addWidget(QLabel("Original View:"))
        side_panel.addWidget(self.input_label)
        side_panel.addWidget(QLabel("Mask View:"))
        side_panel.addWidget(self.mask_label)
        side_panel.addStretch()

        center_layout = QHBoxLayout()
        center_layout.addLayout(side_panel)
        center_layout.addWidget(self.editor, 1, Qt.AlignCenter)

        controls = QHBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.fill_btn)
        controls.addWidget(self.poly_btn)
        controls.addWidget(self.eraser_btn)
        controls.addWidget(QLabel("Brush Size:"))
        controls.addWidget(self.brush_slider)
        controls.addWidget(QLabel("Class:"))
        controls.addWidget(self.class_box)
        
        controls.addStretch()
        self.name_tag = QLabel("Modified by: Gourab Bapli")
        self.name_tag.setStyleSheet("font-weight: bold; color: #444; padding-right: 15px;")
        controls.addWidget(self.name_tag)

        self.date_tag = QLabel(f"Date: 20-Feb-2026")
        self.date_tag.setStyleSheet("font-weight: bold; color: #444; padding-right: 15px;")
        controls.addWidget(self.date_tag)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_name_label)
        main_layout.addLayout(center_layout)
        main_layout.addLayout(controls)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.image_list = []
        self.mask_list = []
        self.current_index = -1

    def search_image(self):   # adding search img
        query = self.search_bar.text().lower()
        if not query or not self.image_list: return
        for i, path in enumerate(self.image_list):
            if query in os.path.basename(path).lower():
                if self.editor.mask_modified: self.save_mask()
                self.current_index = i
                self.load_current_image()
                break

    def load_images(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "Select Start Image")
        mask_path, _ = QFileDialog.getOpenFileName(self, "Select Start Mask")
        if not img_path or not mask_path: return
        img_dir, mask_dir = os.path.dirname(img_path), os.path.dirname(mask_path)
        self.image_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.mask_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if img_path in self.image_list:
            self.current_index = self.image_list.index(img_path)
            self.load_current_image()

    def load_current_image(self):
        if 0 <= self.current_index < len(self.image_list):
            img = cv2.cvtColor(cv2.imread(self.image_list[self.current_index]), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_list[self.current_index], 0)
            self.editor.set_images(img, mask)
            self.show_static(self.input_label, img)
            self.update_mask_view(mask)
            self.image_name_label.setText(f"Image: {os.path.basename(self.image_list[self.current_index])} ({self.current_index+1}/{len(self.image_list)})")

    def load_relative(self, step):
        if not self.image_list: return
        if self.editor.mask_modified: self.save_mask()
        self.current_index = max(0, min(self.current_index + step, len(self.image_list) - 1))
        self.load_current_image()

    def save_mask(self):
        if self.current_index >= 0:
            save_path = self.mask_list[self.current_index]
            cv2.imwrite(save_path, self.editor.mask)
            self.editor.mask_modified = False
            print(f"Saved: {save_path}")

    def toggle_polygon(self, checked):
        self.editor.polygon_mode = checked
        if checked: self.eraser_btn.setChecked(False)

    def toggle_eraser(self, checked):
        self.editor.eraser_mode = checked
        if checked: self.poly_btn.setChecked(False)

    def show_static(self, label, img):
        h, w = img.shape[:2]
        scale = 300 / w if w > 300 else 1
        img_small = cv2.resize(img, (int(w * scale), int(h * scale)))
        if len(img_small.shape) == 2:
            qimg = QImage(img_small.data, img_small.shape[1], img_small.shape[0], img_small.shape[1], QImage.Format_Grayscale8)
            qimg = QImage(img_small.data, img_small.shape[1], img_small.shape[0], img_small.strides[0], QImage.Format_Grayscale8)
        else:
            qimg = QImage(img_small.data, img_small.shape[1], img_small.shape[0], img_small.shape[1] * 3, QImage.Format_RGB888)
            qimg = QImage(img_small.data, img_small.shape[1], img_small.shape[0], img_small.strides[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def update_mask_view(self, mask):
        self.show_static(self.mask_label, mask)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())