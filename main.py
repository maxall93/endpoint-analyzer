import sys
from PyQt6.QtWidgets import QApplication
from gui import DashboardWindow

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName('M365 Endpoint Analyzer')
    app.setStyle('Fusion')  # Use Fusion style for a modern look

    # Create and show the main window
    window = DashboardWindow()
    window.show()

    # Start the application event loop
    sys.exit(app.exec())

if __name__ == '__main__':
    main()