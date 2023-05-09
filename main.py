import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QScrollArea,
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

        # Create grid layout
        grid_layout = QGridLayout()

        # Add grid layout to main layout
        layout.addLayout(grid_layout)

        # Create central widget and set layout
        self.scroll = QScrollArea()
        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(central_widget)

        # Set central widget
        self.setCentralWidget(self.scroll)

        # Set window properties
        self.setWindowTitle("Image Segmentation")
        self.resize(800, 600)

        self.widget_1 = MaskImage()
        grid_layout.addWidget(self.widget_1)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
