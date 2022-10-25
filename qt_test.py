import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QMainWindow

app = QApplication(sys.argv)

def buttonPushed():
	print('button pushed!')

window = QMainWindow()
btn = QPushButton("Push me")
window.setCentralWidget(btn)
btn.clicked.connect(buttonPushed)
window.show()

app.exec()
