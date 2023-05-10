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
)
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from imagefilter import MaskImage


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


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
