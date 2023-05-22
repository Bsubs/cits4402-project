import sys
import json
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QScrollArea,
    QTabWidget,
    QPushButton,
    QLabel,
    QGridLayout
)
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from segmentimage import MaskImage
from aligncameras import TriangulateImage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_gui()

    
    def init_gui(self):
        # Create layout for widgets
        layout = QVBoxLayout()

        # Create tab widget 
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Initialize dictionary to hold all the segmentimage widgets
        self.widgets = {}

        # Read data from file and create tabs
        with open('tuned_hyperparameters.json', 'r') as f:
            data = json.load(f)
        for name, params in data.items():
            tab = QWidget()
            tab_layout = QVBoxLayout()
            widget = MaskImage(param_dict=params)
            tab_layout.addWidget(widget)
            tab.setLayout(tab_layout)
            self.tab_widget.addTab(tab, name)
            self.widgets[name] = widget
                
        # Create the tab for 3D rendering
        tab_3D = QWidget()
        tab_layout_3D = QVBoxLayout()
        self.triangulate_widget = TriangulateImage(widget_dict=self.widgets)
        tab_layout_3D.addWidget(self.triangulate_widget)
        tab_3D.setLayout(tab_layout_3D)
        self.tab_widget.addTab(tab_3D, '3D Render of Room')

        # Create the tab for Run all
        tab_run_all = QWidget()
        tab_layout_run_all = QGridLayout()
        run_all_btn = QPushButton("Perform Image Segmentation for all Cameras", self)
        run_all_btn.clicked.connect(self.run_all_images)
        tab_layout_run_all.addWidget(run_all_btn, 1, 0)
        self.loading_label = QLabel("")
        self.current_label = QLabel("")
        tab_layout_run_all.addWidget(self.loading_label, 2, 0)
        tab_layout_run_all.addWidget(self.current_label, 3, 0)
        tab_run_all.setLayout(tab_layout_run_all)
        self.tab_widget.insertTab(0, tab_run_all, 'Perform Image Segmentation')
        
        # Select the first tab
        self.tab_widget.setCurrentIndex(0)
        # Create central widget and set layout
        central_widget = QWidget()
        central_widget.setLayout(layout)

        # Create scroll area
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(central_widget)

        # Set central widget
        self.setCentralWidget(self.scroll)

        # Set window properties
        self.setWindowTitle("Image Segmentation")
        self.resize(800, 600)

    def run_all_images(self):
        self.set_loading_text("Running image segmentation, please wait patiently")
        QApplication.processEvents()
        for i, widget in enumerate(self.widgets):
            self.current_label.setText("Currently Segmenting: " + widget)
            QApplication.processEvents()
            self.widgets[widget].run_all()

        self.set_loading_text("Image Segmentation Completed.")
        self.current_label.setText(" ")
    
    def set_loading_text(self, text):
        self.loading_label.setText(text)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
