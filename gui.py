import os
import sys
import logging
from datetime import datetime, timedelta
import time
import random
import collections
import json
from pathlib import Path
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QSplitter, QTextEdit, QMessageBox, 
    QStatusBar, QSizePolicy, QScrollArea, QGroupBox, QGridLayout,
    QFrame, QSpacerItem, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QColor, QPalette

import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for PyQt6 compatibility
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from service_checker import ServiceChecker

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'gui_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will also print to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting application, logging to {log_file}")

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PyQt6 application"""
    def __init__(self, parent=None, width=5, height=4, dpi=100, dark_mode=True):
        # Create figure with dark mode style if requested
        if dark_mode:
            plt.style.use('dark_background')
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        
        # Basic styling
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Make canvas expandable - fixed to use QSizePolicy instead of QHeaderView.Policy
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)
        
    def clear_plot(self):
        self.axes.clear()
        self.draw()


class LatencyGraph(QWidget):
    """Widget to display latency graph for a single endpoint"""
    def __init__(self, endpoint_name, service_checker, parent=None, auto_generate_test_data=True):
        super().__init__(parent)
        self.endpoint_name = endpoint_name
        self.service_checker = service_checker
        self.auto_generate_test_data = auto_generate_test_data
        self.setMinimumHeight(250)
        
        logger.info(f"Creating LatencyGraph for {endpoint_name}")
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create header with endpoint name
        header_layout = QHBoxLayout()
        header = QLabel(self.endpoint_name)
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        port_label = QLabel("Port: 443")
        port_label.setStyleSheet("color: #aaaaaa;")
        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(port_label)
        layout.addLayout(header_layout)
        
        # Create matplotlib canvas
        self.canvas = MatplotlibCanvas(self, width=5, height=3, dpi=100)
        layout.addWidget(self.canvas)
        
        # Status indicators at the bottom of the graph
        bottom_layout = QHBoxLayout()
        
        # Average latency range indicator (left)
        self.avg_range_label = QLabel("Avg Range: -- ms")
        self.avg_range_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        bottom_layout.addWidget(self.avg_range_label)
        
        bottom_layout.addStretch()
        
        # Status indicator (right)
        self.status_label = QLabel("Stable")
        self.status_label.setStyleSheet("background-color: #4CAF50; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;")
        bottom_layout.addWidget(self.status_label)
        
        layout.addLayout(bottom_layout)
        
        # Initialize with empty data
        self.has_alert = False
        self.initialize_plot()
        
    def initialize_plot(self):
        """Initialize the plot with empty data"""
        self.canvas.axes.clear()
        self.canvas.axes.set_facecolor('#2d2d2d')
        self.canvas.axes.grid(True, linestyle='--', alpha=0.3)
        
        # Set labels and format
        self.canvas.axes.set_ylabel('Latency (ms)')
        self.canvas.axes.set_xlabel('Time')
        
        # Y-axis range
        self.canvas.axes.set_ylim(0, 250)
        
        # Alert threshold line
        self.canvas.axes.axhline(y=250, color='#ff5252', linestyle='-', alpha=0.5)
        
        # Formatting - use fig.autofmt_xdate() instead of axes.autofmt_xdate()
        self.canvas.fig.autofmt_xdate()
        self.canvas.draw()
        
    def get_endpoint_with_port(self, endpoint_name):
        """Convert a domain name to the 'domain:port' format used in latency_history"""
        # Common port mappings
        return f"{endpoint_name}:443"  # Default to 443
        
    def get_service_for_endpoint(self, endpoint_name):
        """Determine which service an endpoint belongs to"""
        endpoint_service_map = {
            'teams.microsoft.com': 'Microsoft Teams',
            'presence.teams.microsoft.com': 'Microsoft Teams',
            'outlook.office365.com': 'Exchange Online',
            'outlook.office.com': 'Exchange Online',
            'smtp.office365.com': 'Exchange Online',
            'sharepoint.com': 'SharePoint & OneDrive',
            'onedrive.com': 'SharePoint & OneDrive',
            'graph.microsoft.com': 'Microsoft Graph',
            'login.microsoftonline.com': 'Microsoft Graph'
        }
        
        # Check direct match
        if endpoint_name in endpoint_service_map:
            return endpoint_service_map[endpoint_name]
            
        # Check partial match
        for key, service in endpoint_service_map.items():
            if key in endpoint_name or endpoint_name in key:
                return service
                
        # Try to infer service based on domain parts
        if 'teams' in endpoint_name:
            return 'Microsoft Teams'
        elif 'outlook' in endpoint_name or 'exchange' in endpoint_name or 'office365' in endpoint_name:
            return 'Exchange Online'
        elif 'sharepoint' in endpoint_name or 'onedrive' in endpoint_name:
            return 'SharePoint & OneDrive'
        elif 'graph' in endpoint_name:
            return 'Microsoft Graph'
        elif 'login' in endpoint_name or 'microsoftonline' in endpoint_name:
            return 'Microsoft Graph'  # Login endpoints are categorized under Graph
            
        # Default to unknown
        return None
        
    def find_latency_data(self):
        """Find the correct latency data for this endpoint"""
        if not hasattr(self.service_checker, 'latency_history') or not self.service_checker.latency_history:
            logger.warning(f"No latency_history available in service_checker")
            return None
            
        # Try direct endpoint name
        if self.endpoint_name in self.service_checker.latency_history:
            data = self.service_checker.latency_history[self.endpoint_name]
            logger.debug(f"Found direct endpoint match: {self.endpoint_name}")
            return data
            
        # Try with endpoint:port format
        endpoint_with_port = self.get_endpoint_with_port(self.endpoint_name)
        
        # Try to get the service for this endpoint
        service_name = self.get_service_for_endpoint(self.endpoint_name)
        
        if service_name and service_name in self.service_checker.latency_history:
            # Check if endpoint:port exists in this service
            if endpoint_with_port in self.service_checker.latency_history[service_name]:
                logger.debug(f"Found endpoint via service: {service_name} -> {endpoint_with_port}")
                return self.service_checker.latency_history[service_name][endpoint_with_port]
                
            # Check for substring matches
            for endpoint_key in self.service_checker.latency_history[service_name]:
                if (self.endpoint_name in endpoint_key) or (endpoint_key in self.endpoint_name):
                    logger.debug(f"Found endpoint via substring: {service_name} -> {endpoint_key}")
                    return self.service_checker.latency_history[service_name][endpoint_key]
        
        # Try to find any key containing this endpoint name
        for service, endpoints in self.service_checker.latency_history.items():
            for endpoint_key in endpoints:
                if isinstance(endpoint_key, str) and (self.endpoint_name in endpoint_key or endpoint_key in self.endpoint_name):
                    logger.debug(f"Found endpoint via global search: {service} -> {endpoint_key}")
                    return endpoints[endpoint_key]
                    
        logger.warning(f"Could not find latency data for {self.endpoint_name}")
        return None

    def update_plot(self, time_window_minutes=15):
        """Update the plot with the latest data"""
        try:
            # Find latency data for this endpoint
            history = self.find_latency_data()
            
            if not history:
                logger.warning(f"No latency data found for {self.endpoint_name}")
                print(f"No latency data found for {self.endpoint_name}")
                # Clear the plot and show "No Data"
                self.initialize_plot()
                self.canvas.axes.text(0.5, 0.5, "No Data", 
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=self.canvas.axes.transAxes,
                                    fontsize=14, color='#aaaaaa')
                self.canvas.draw()
                return
            
            # Clear the plot
            self.canvas.axes.clear()
            
            # Set up the plot
            self.canvas.axes.set_facecolor('#212121')
            self.canvas.axes.set_ylabel('Latency (ms)')
            self.canvas.axes.set_xlabel('Time')
            self.canvas.axes.grid(True, linestyle='--', alpha=0.7)
            
            # Process the data
            times = []
            latencies = []
            
            # Get current time for filtering
            now = datetime.now()
            cutoff_time = now - timedelta(minutes=time_window_minutes)
            
            # Determine data format and process accordingly
            if history and isinstance(history, dict):
                logger.debug(f"Processing dictionary data format for {self.endpoint_name}")
                # Dictionary format - extract timestamp-latency pairs
                for timestamp, latency in history.items():
                    try:
                        # Convert timestamp to datetime if it's a string
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        elif isinstance(timestamp, (int, float)):
                            dt = datetime.fromtimestamp(timestamp)
                        else:
                            dt = timestamp
                            
                        # Only include data points within the time window
                        if dt >= cutoff_time:
                            times.append(dt)
                            latencies.append(float(latency))
                    except Exception as e:
                        logger.warning(f"Error processing data point {timestamp}: {e}")
            
            elif history and isinstance(next(iter(history), None), tuple):
                logger.debug(f"Processing tuple data format for {self.endpoint_name}")
                # Tuple format (timestamp, latency)
                for timestamp, latency in history:
                    try:
                        # Convert timestamp to datetime if it's a string
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        elif isinstance(timestamp, (int, float)):
                            dt = datetime.fromtimestamp(timestamp)
                        else:
                            dt = timestamp
                            
                        # Only include data points within the time window
                        if dt >= cutoff_time:
                            times.append(dt)
                            latencies.append(float(latency))
                    except Exception as e:
                        logger.warning(f"Error processing data point {timestamp}: {e}")
            
            else:
                logger.debug(f"Processing raw values format for {self.endpoint_name}")
                # Raw list of values (create synthetic timestamps)
                for i, latency in enumerate(history):
                    try:
                        # Create synthetic timestamps going backward from now
                        dt = now - timedelta(seconds=(len(history) - i - 1) * 15)
                        times.append(dt)
                        latencies.append(float(latency))
                    except Exception as e:
                        logger.warning(f"Error processing data point {i}: {e}")
            
            # If we have no data points after processing, show "No Data"
            if not times or not latencies:
                logger.warning(f"No valid data points for {self.endpoint_name} after processing")
                self.canvas.axes.text(0.5, 0.5, "No Data", 
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=self.canvas.axes.transAxes,
                                    fontsize=14, color='#aaaaaa')
                self.canvas.draw()
                return
            
            # Calculate average, min, and max latency for shading
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Create the plot
            self.canvas.axes.plot(times, latencies, '-o', color='#2196F3', linewidth=2, label='Latency', 
                                markersize=4, markerfacecolor='white', markeredgecolor='#2196F3')
            
            # Add shaded area for min/max range
            self.canvas.axes.fill_between(times, min_latency, max_latency, alpha=0.2, color='#4CAF50', label='Min/Max Range')
            
            # Add baseline fill
            self.canvas.axes.fill_between(times, 0, latencies, alpha=0.1, color='#2196F3')
            
            # Add threshold line at 250ms
            self.canvas.axes.axhline(y=250, color='#ff5252', linestyle='-', alpha=0.5, label='Alert Threshold')
            
            # Set limits
            self.canvas.axes.set_ylim(0, max(max(latencies) * 1.2, 250))  # At least show up to 250ms
            if times:
                self.canvas.axes.set_xlim(min(times), max(times))
            
            # Add legend
            self.canvas.axes.legend(loc='upper right', fontsize='small')
            
            # Format the date axis
            self.canvas.fig.autofmt_xdate()
            
            # Update the plot
            self.canvas.draw()
            
            # Update the average range label
            self.avg_range_label.setText(f"Avg Range: {int(min_latency)}-{int(max_latency)} ms")
            
            # Update stability indicator
            stability = "Stable" if max_latency - min_latency < 50 else "Variable"
            self.status_label.setText(stability)
            if stability == "Stable":
                self.status_label.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px; padding: 2px 8px;")
            else:
                self.status_label.setStyleSheet("background-color: #FF9800; color: white; border-radius: 10px; padding: 2px 8px;")
            
            # Add detailed stats text in bottom left if service_checker has the method
            if hasattr(self.service_checker, 'get_latency_stats'):
                try:
                    # Get endpoint with port if available
                    endpoint_with_port = self.get_endpoint_with_port(self.endpoint_name)
                    
                    # Get latency stats from service checker
                    stats = self.service_checker.get_latency_stats(endpoint_with_port)
                    
                    if stats['has_data']:
                        # Format the stats text
                        stats_text = (
                            f"Avg: {stats['avg']:.1f} ms\n"
                            f"Min: {stats['min']:.1f} ms\n"
                            f"Max: {stats['max']:.1f} ms\n"
                            f"Current: {stats['current']:.1f} ms"
                        )
                        
                        # Add text to bottom left of plot
                        self.canvas.axes.text(
                            0.02, 0.02,  # Position (2% from left, 2% from bottom)
                            stats_text,
                            transform=self.canvas.axes.transAxes,
                            fontsize=9,
                            verticalalignment='bottom',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', alpha=0.7, edgecolor='#555555')
                        )
                        self.canvas.draw()  # Redraw to show the text
                except Exception as e:
                    logger.error(f"Error adding stats text: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error updating plot for {self.endpoint_name}: {str(e)}")
            traceback.print_exc()
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, "Error updating plot", 
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=self.canvas.axes.transAxes,
                                fontsize=12, color='#ff5252')
            self.canvas.draw()

    def generate_test_data(self, time_window_minutes=15):
        """Generate test data for this endpoint on the fly"""
        logger.info(f"Generating test data for {self.endpoint_name}")
        print(f"Generating test data for {self.endpoint_name}")
        
        if not hasattr(self.service_checker, 'latency_history'):
            self.service_checker.latency_history = {}
            
        # Create or update the endpoint in latency_history
        if self.endpoint_name not in self.service_checker.latency_history:
            self.service_checker.latency_history[self.endpoint_name] = {}
        
        # Generate test data points for the time window
        current_time = datetime.now()
        test_latency = []
        points_count = time_window_minutes * 4  # 15s intervals
        for i in range(points_count):
            point_time = current_time - timedelta(seconds=15*i)
            # Generate random latency between 40-80ms with some spikes
            latency = 50 + 30 * (0.5 - 0.5 * (i % 10 == 0))  # Spike every 10th point
            test_latency.append((point_time, latency))
        
        # Create a collections.deque for the test data
        # Match the format used by ServiceChecker: endpoint -> protocol -> deque of data points
        self.service_checker.latency_history[self.endpoint_name]['HTTPS_443'] = collections.deque(test_latency, maxlen=240)
        
        logger.info(f"Generated {len(test_latency)} test data points for {self.endpoint_name}")
        print(f"Generated {len(test_latency)} test data points for {self.endpoint_name}")
        
        # Plot the test data directly instead of recursively calling update_plot
        self.canvas.axes.clear()
        self.canvas.axes.set_facecolor('#2d2d2d')
        self.canvas.axes.grid(True, linestyle='--', alpha=0.3)
        
        # Set labels and format
        self.canvas.axes.set_ylabel('Latency (ms)')
        self.canvas.axes.set_xlabel('Time')
        
        # Y-axis range
        self.canvas.axes.set_ylim(0, 250)
        
        # Alert threshold line
        self.canvas.axes.axhline(y=250, color='#ff5252', linestyle='-', alpha=0.5)
        
        # Convert data for plotting
        times = [point[0] for point in test_latency]
        latencies = [point[1] for point in test_latency]
        
        # Create the plot with test data
        self.canvas.axes.plot(times, latencies, '-', color='#2196F3', linewidth=2, label='Latency')
        self.canvas.axes.fill_between(times, 0, latencies, alpha=0.2, color='#2196F3')
        
        # Format date axis
        self.canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Add padding so the line doesn't touch the edges
        if times:
            self.canvas.axes.set_xlim(min(times) - timedelta(minutes=1), max(times) + timedelta(minutes=1))
        
        # Update status labels
        self.avg_range_label.setText(f"Avg Range: 50-80 ms (Test Data)")
        self.status_label.setText("Test Data")
        self.status_label.setStyleSheet("background-color: #FFA500; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;")
        
        # Draw the plot
        self.canvas.fig.autofmt_xdate()
        self.canvas.draw()
        print(f"Test data plot generated for {self.endpoint_name}")


class ServiceCheckerThread(QThread):
    """Thread for running service checks in the background"""
    update_signal = pyqtSignal(dict)
    
    def __init__(self, service_checker):
        super().__init__()
        self.service_checker = service_checker
        self.running = True
        
    def run(self):
        while self.running:
            # Run service checks - this actually performs the checks and populates data
            results = self.service_checker.run_service_checks()
            
            # Emit signal with results
            self.update_signal.emit(results)
            
            # Wait before next check
            self.msleep(15000)  # 15 seconds between checks
            
    def stop(self):
        self.running = False
        self.wait()


class DashboardWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Initialize ServiceChecker
        self.service_checker = ServiceChecker()
        
        # Set up UI
        self.setWindowTitle("M365 Endpoint Analyzer")
        self.resize(1200, 800)
        self.setup_dark_theme()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Set up tabs
        self.setup_overview_tab()
        self.setup_detailed_status_tab()
        self.setup_latency_trends_tab()
        
        # Add refresh button in top right corner
        header_layout = QHBoxLayout()
        header_layout.addStretch()
        
        # Time interval selection combo box
        time_interval_label = QLabel("Time Interval:")
        time_interval_label.setStyleSheet("color: #dddddd;")
        self.time_interval_combo = QComboBox()
        self.time_interval_combo.addItems(["5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"])
        self.time_interval_combo.setCurrentIndex(1)  # Default to 15 minutes
        self.time_interval_combo.currentIndexChanged.connect(self.update_time_interval)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_data)
        
        header_layout.addWidget(time_interval_label)
        header_layout.addWidget(self.time_interval_combo)
        header_layout.addWidget(refresh_button)
        
        main_layout.insertLayout(0, header_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Setup update timer and thread first
        self.setup_checker_thread()
        
        # Run initial checks to populate data after thread is started
        self.initialize_data()
        
    def setup_dark_theme(self):
        """Set up dark theme for the application"""
        dark_palette = QPalette()
        
        # Base colors
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(212, 212, 212))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(212, 212, 212))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(212, 212, 212))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(212, 212, 212))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        # Apply the dark theme
        self.setPalette(dark_palette)
        
        # Stylesheet for custom styling
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2d2d2d;
                color: #d4d4d4;
            }
            
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #2d2d2d;
            }
            
            QTabBar::tab {
                background-color: #1e1e1e;
                color: #d4d4d4;
                padding: 8px 16px;
                border: 1px solid #3d3d3d;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                border-bottom-color: #2d2d2d;
            }
            
            QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #3d3d3d;
                outline: none;
            }
            
            QTableWidget::item {
                padding: 6px;
            }
            
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #d4d4d4;
                padding: 6px;
                border: 1px solid #3d3d3d;
            }
            
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            
            QPushButton:hover {
                background-color: #0063b1;
            }
            
            QPushButton:pressed {
                background-color: #004b8c;
            }
            
            QComboBox {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                padding: 4px 8px;
                border-radius: 4px;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #3d3d3d;
                border-left-style: solid;
            }
            
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                margin: 0px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #5d5d5d;
                min-height: 20px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #7d7d7d;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QLabel {
                color: #d4d4d4;
            }
        """)
        
    def setup_overview_tab(self):
        """Set up the overview tab"""
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # Create table for service overview
        self.overview_table = QTableWidget(0, 3)
        self.overview_table.setHorizontalHeaderLabels(['Service', 'Status', 'Details'])
        self.overview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.overview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.overview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.overview_table.verticalHeader().setVisible(False)
        
        overview_layout.addWidget(self.overview_table)
        
        # Add overview tab to main tab widget
        self.tabs.addTab(overview_tab, "Service Overview")
        
    def setup_detailed_status_tab(self):
        """Set up the detailed status tab"""
        detailed_tab = QWidget()
        detailed_layout = QVBoxLayout(detailed_tab)
        
        # Create table for detailed endpoint status
        self.detailed_table = QTableWidget(0, 6)
        self.detailed_table.setHorizontalHeaderLabels(['Status', 'Service', 'Endpoint', 'Protocol', 'Port', 'Error Details'])
        self.detailed_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.detailed_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # Error details column
        self.detailed_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.detailed_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.detailed_table.verticalHeader().setVisible(False)
        
        detailed_layout.addWidget(self.detailed_table)
        
        # Add detailed tab to main tab widget
        self.tabs.addTab(detailed_tab, "Detailed Status")
        
    def setup_latency_trends_tab(self):
        """Set up the latency trends tab"""
        try:
            logger.info("Setting up latency trends tab")
            print("Creating latency trends tab...")
            
            # Create the tab widget
            trends_tab = QWidget()
            trends_layout = QVBoxLayout(trends_tab)
            
            # Create scroll area for graphs
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_contents = QWidget()
            scroll_layout = QGridLayout(scroll_contents)
            
            # Define the core endpoints we want to monitor (all using HTTPS port 443)
            self.core_endpoints = [
                'teams.microsoft.com',
                'presence.teams.microsoft.com',
                'outlook.office365.com',
                'sharepoint.com',
                'graph.microsoft.com',
                'login.microsoftonline.com'
            ]
            
            # Create graph widgets for each core endpoint
            self.graph_widgets = {}
            row, col = 0, 0
            for endpoint in self.core_endpoints:
                try:
                    logger.debug(f"Creating graph for {endpoint}")
                    print(f"Creating graph for {endpoint}")
                    # Set auto_generate_test_data to False to ensure we only use real data
                    graph = LatencyGraph(endpoint, self.service_checker, parent=scroll_contents, auto_generate_test_data=False)
                    self.graph_widgets[endpoint] = graph
                    scroll_layout.addWidget(graph, row, col)
                    
                    # Update grid position
                    col += 1
                    if col > 1:  # Two columns of graphs
                        col = 0
                        row += 1
                except Exception as e:
                    logger.error(f"Error creating graph for {endpoint}: {str(e)}", exc_info=True)
                    print(f"Error creating graph for {endpoint}: {str(e)}")
            
            # Set the scroll contents
            scroll_contents.setLayout(scroll_layout)
            scroll_area.setWidget(scroll_contents)
            
            # Add scroll area to the main layout
            trends_layout.addWidget(scroll_area)
            
            # Add tab to the main tab widget
            self.tabs.addTab(trends_tab, "Latency Trends")
            
            logger.info(f"Created {len(self.graph_widgets)} graph widgets")
            print(f"Created {len(self.graph_widgets)} graph widgets")
        except Exception as e:
            logger.error(f"Error setting up latency trends tab: {str(e)}", exc_info=True)
            print(f"Error setting up latency trends tab: {str(e)}")
            
            # Create a simple error message tab
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_label = QLabel("Error loading latency graphs. Check logs for details.")
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            self.tabs.addTab(error_tab, "Latency Trends")
            
    def setup_checker_thread(self):
        """Set up the background thread for service checking"""
        self.checker_thread = ServiceCheckerThread(self.service_checker)
        self.checker_thread.update_signal.connect(self.update_ui)
        self.checker_thread.start()
        
    def update_ui(self, results):
        """Update the UI with the latest check results"""
        # Update status bar
        self.statusBar().showMessage(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update Overview tab
        self.update_overview_table(results)
        
        # Update Detailed Status tab
        self.update_detailed_table(results)
        
        # Update Latency Trends tab
        self.update_latency_graphs()
        
    def update_overview_table(self, results):
        """Update the overview table with service status summary"""
        if hasattr(self.service_checker, 'get_service_status'):
            service_status = self.service_checker.get_service_status(results)
        else:
            # Fallback if method doesn't exist - create basic status
            service_status = {}
            for service_name in results.keys():
                service_status[service_name] = {
                    'status': 'Unknown',
                    'details': 'Service status information not available'
                }
        
        # Clear table
        self.overview_table.setRowCount(0)
        
        # Add rows for each service
        for service_name, status in service_status.items():
            row_position = self.overview_table.rowCount()
            self.overview_table.insertRow(row_position)
            
            # Service Name
            service_item = QTableWidgetItem(service_name)
            self.overview_table.setItem(row_position, 0, service_item)
            
            # Status
            status_text = status.get('status', 'Unknown') if isinstance(status, dict) else str(status)
            status_item = QTableWidgetItem(status_text)
            
            if isinstance(status, dict) and status.get('status') == 'Healthy':
                status_item.setBackground(QColor('#4CAF50'))
                status_item.setForeground(QColor('white'))
            else:
                status_item.setBackground(QColor('#F44336'))
                status_item.setForeground(QColor('white'))
            self.overview_table.setItem(row_position, 1, status_item)
            
            # Details
            details_text = status.get('details', '') if isinstance(status, dict) else ''
            details_item = QTableWidgetItem(details_text)
            self.overview_table.setItem(row_position, 2, details_item)
            
    def update_detailed_table(self, results):
        """Update the detailed status table with all endpoint checks"""
        # Clear table
        self.detailed_table.setRowCount(0)
        
        # Debug info
        logger.info(f"Updating detailed table with results: {list(results.keys())}")
        
        # Group unhealthy services at the top
        unhealthy_rows = []
        healthy_rows = []
        
        # First try the expected structure (endpoints with ports)
        rows_added = False
        
        # Process results into row data
        for service_name, service_data in results.items():
            if not isinstance(service_data, dict):
                logger.warning(f"Warning: Service data for {service_name} is not a dictionary")
                continue
                
            logger.debug(f"Processing service: {service_name}, keys: {service_data.keys()}")
            
            # Check if the expected 'endpoints' structure exists
            if 'endpoints' in service_data:
                for endpoint_data in service_data.get('endpoints', []):
                    if not isinstance(endpoint_data, dict):
                        logger.warning(f"Warning: Endpoint data in {service_name} is not a dictionary")
                        continue
                        
                    domain = endpoint_data.get('domain', '')
                    logger.debug(f"  Processing endpoint: {domain}, keys: {endpoint_data.keys()}")
                    
                    for port_result in endpoint_data.get('ports', []):
                        if not isinstance(port_result, dict):
                            logger.warning(f"Warning: Port result for {domain} is not a dictionary")
                            continue
                            
                        protocol = port_result.get('protocol', '')
                        port = port_result.get('port', '')
                        is_healthy = port_result.get('is_healthy', False)
                        error = port_result.get('error', '')
                        
                        logger.debug(f"    Processing port: {port}/{protocol}, healthy: {is_healthy}")
                        
                        row_data = [
                            is_healthy,  # Status icon is determined by this boolean
                            service_name,
                            domain,
                            protocol,
                            str(port),
                            error
                        ]
                        
                        if is_healthy:
                            healthy_rows.append(row_data)
                        else:
                            unhealthy_rows.append(row_data)
                        
                rows_added = len(healthy_rows) > 0 or len(unhealthy_rows) > 0
            
            # Alternative format: Check for dns_checks, port_checks, http_checks
            if not rows_added and all(k in service_data for k in ['dns_checks', 'port_checks', 'http_checks']):
                logger.info(f"Trying alternative format for {service_name}")
                
                # Add DNS checks
                for dns_check in service_data.get('dns_checks', []):
                    endpoint = dns_check.get('endpoint', '')
                    result = dns_check.get('result', {})
                    is_healthy = result.get('success', False)
                    error = result.get('error', '')
                    
                    row_data = [
                        is_healthy,
                        service_name,
                        endpoint,
                        'DNS',
                        '53',
                        error
                    ]
                    
                    logger.debug(f"  DNS check for {endpoint}: {is_healthy}")
                    
                    if is_healthy:
                        healthy_rows.append(row_data)
                    else:
                        unhealthy_rows.append(row_data)
                
                # Add port checks
                for port_check in service_data.get('port_checks', []):
                    endpoint = port_check.get('endpoint', '')
                    result = port_check.get('result', {})
                    is_healthy = result.get('success', False)
                    error = result.get('error', '')
                    
                    # Extract port and protocol from endpoint
                    try:
                        domain, port_str = endpoint.split(':')
                        protocol = 'TCP'  # Default
                        
                        # Try to determine protocol from port
                        port_protocols = {
                            '443': 'HTTPS',
                            '80': 'HTTP',
                            '25': 'SMTP',
                            '587': 'SMTP-TLS',
                            '993': 'IMAP',
                            '995': 'POP3'
                        }
                        if port_str in port_protocols:
                            protocol = port_protocols[port_str]
                    except:
                        domain = endpoint
                        port_str = '0'
                        protocol = 'Unknown'
                    
                    row_data = [
                        is_healthy,
                        service_name,
                        domain,
                        protocol,
                        port_str,
                        error
                    ]
                    
                    logger.debug(f"  Port check for {endpoint}: {is_healthy}")
                    
                    if is_healthy:
                        healthy_rows.append(row_data)
                    else:
                        unhealthy_rows.append(row_data)
                
                # Add HTTP checks
                for http_check in service_data.get('http_checks', []):
                    endpoint = http_check.get('endpoint', '')
                    result = http_check.get('result', {})
                    is_healthy = result.get('success', False)
                    error = result.get('error', '')
                    status_code = result.get('status_code', '')
                    
                    if status_code:
                        detail = f"Status: {status_code}"
                        if error:
                            detail += f", Error: {error}"
                    else:
                        detail = error
                    
                    row_data = [
                        is_healthy,
                        service_name,
                        endpoint,
                        'HTTP',
                        '80/443',
                        detail
                    ]
                    
                    logger.debug(f"  HTTP check for {endpoint}: {is_healthy}")
                    
                    if is_healthy:
                        healthy_rows.append(row_data)
                    else:
                        unhealthy_rows.append(row_data)
        
        # If no data found with either format, add test data
        if len(healthy_rows) == 0 and len(unhealthy_rows) == 0:
            logger.warning("No data extracted from results, adding test data")
            test_results = self.create_test_results()
            for service_name, service_data in test_results.items():
                for endpoint_data in service_data.get('endpoints', []):
                    domain = endpoint_data.get('domain', '')
                    for port_result in endpoint_data.get('ports', []):
                        protocol = port_result.get('protocol', '')
                        port = port_result.get('port', '')
                        is_healthy = port_result.get('is_healthy', False)
                        error = port_result.get('error', '')
                        
                        row_data = [
                            is_healthy,  # Status icon is determined by this boolean
                            service_name,
                            domain,
                            protocol,
                            str(port),
                            error
                        ]
                        
                        if is_healthy:
                            healthy_rows.append(row_data)
                        else:
                            unhealthy_rows.append(row_data)
        
        # Add unhealthy rows first, then healthy ones
        all_rows = unhealthy_rows + healthy_rows
        
        logger.info(f"Total rows to display: {len(all_rows)} ({len(unhealthy_rows)} unhealthy, {len(healthy_rows)} healthy)")
        
        # Populate the table
        for row_data in all_rows:
            row_position = self.detailed_table.rowCount()
            self.detailed_table.insertRow(row_position)
            
            # Status icon (green checkmark or red X)
            status_label = QLabel()
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            if row_data[0]:  # is_healthy
                status_label.setStyleSheet("background-color: #4CAF50; border-radius: 10px;")
                status_label.setText("✓")
            else:
                status_label.setStyleSheet("background-color: #F44336; border-radius: 10px;")
                status_label.setText("✗")
                
            self.detailed_table.setCellWidget(row_position, 0, status_label)
            
            # Other columns
            for col in range(1, len(row_data)):
                self.detailed_table.setItem(row_position, col, QTableWidgetItem(str(row_data[col])))
                
        logger.info(f"Detailed table updated with {self.detailed_table.rowCount()} rows")
    
    def update_latency_graphs(self):
        """Update all latency graphs"""
        # Get selected time interval
        time_interval = self.get_selected_time_interval()
        
        logger.info(f"Updating latency graphs with time interval: {time_interval} minutes")
        print(f"Updating {len(self.graph_widgets) if hasattr(self, 'graph_widgets') else 0} latency graphs with time interval: {time_interval} minutes")
        logger.debug(f"Graph widgets: {list(self.graph_widgets.keys()) if hasattr(self, 'graph_widgets') else 'None'}")
        
        # Make sure we have graph widgets
        if not hasattr(self, 'graph_widgets') or not self.graph_widgets:
            logger.warning("No graph widgets found - rebuilding graphs")
            print("No graph widgets found - rebuilding graphs")
            
            # Try to recreate the latency tab
            try:
                if hasattr(self, 'tabs'):
                    # Find and remove any existing latency trends tab
                    for i in range(self.tabs.count()):
                        if self.tabs.tabText(i) == "Latency Trends":
                            self.tabs.removeTab(i)
                            break
                    
                    # Recreate the latency trends tab
                    self.setup_latency_trends_tab()
                else:
                    logger.error("Cannot rebuild graphs - tabs widget not found")
            except Exception as e:
                logger.error(f"Error rebuilding latency graphs: {str(e)}", exc_info=True)
                print(f"Error rebuilding latency graphs: {str(e)}")
            return
        
        # Log the structure of the latency_history
        if hasattr(self.service_checker, 'latency_history'):
            logger.debug(f"latency_history structure: {self.service_checker.latency_history.keys()}")
            try:
                for domain in self.service_checker.latency_history:
                    if isinstance(self.service_checker.latency_history[domain], dict):
                        logger.debug(f"  {domain} protocols: {self.service_checker.latency_history[domain].keys()}")
            except Exception as e:
                logger.error(f"Error examining latency_history: {str(e)}")
        else:
            logger.warning("service_checker has no latency_history attribute")
        
        # Update each graph
        updated_count = 0
        for endpoint, graph in self.graph_widgets.items():
            try:
                print(f"Updating graph for {endpoint}")
                graph.update_plot(time_window_minutes=time_interval)
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating graph for {endpoint}: {str(e)}", exc_info=True)
                print(f"Error updating graph for {endpoint}: {str(e)}")
        
        logger.info(f"Updated {updated_count} of {len(self.graph_widgets)} graphs")
        print(f"Updated {updated_count} of {len(self.graph_widgets)} graphs")
        
    def update_time_interval(self):
        """Handle time interval change"""
        time_interval = self.get_selected_time_interval()
        
        # Update all graphs with new time interval
        for endpoint, graph in self.graph_widgets.items():
            graph.update_plot(time_window_minutes=time_interval)
            
    def get_selected_time_interval(self):
        """Get the currently selected time interval in minutes"""
        selection = self.time_interval_combo.currentText()
        if selection == "5 Minutes":
            return 5
        elif selection == "15 Minutes":
            return 15
        elif selection == "30 Minutes":
            return 30
        elif selection == "1 Hour":
            return 60
        return 15  # Default
            
    def refresh_data(self):
        """Manually refresh data"""
        results = self.service_checker.run_service_checks()
        self.update_ui(results)
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the checker thread when closing
        if hasattr(self, 'checker_thread'):
            self.checker_thread.stop()
        event.accept()

    def initialize_data(self):
        """Run initial checks to populate data"""
        try:
            logger.info("Initializing data...")
            
            # First, make sure the service_checker knows about the endpoints we want to monitor
            if not hasattr(self.service_checker, 'endpoints') or not self.service_checker.endpoints:
                # Log warning about missing endpoints
                logger.warning("Warning: No endpoints defined in service_checker")
            else:
                logger.info(f"Found {len(self.service_checker.endpoints)} service endpoints defined")
            
            # Run initial service checks to populate data
            try:
                if hasattr(self.service_checker, 'run_service_checks'):
                    logger.info("Running service checks...")
                    results = self.service_checker.run_service_checks()
                    logger.info(f"Service checks completed. Results: {len(results)} services")
                    
                    # Wait up to 20 seconds for the checker thread to collect some real data
                    wait_time = 0
                    while wait_time < 20:
                        if any(len(history) > 0 for history in self.service_checker.latency_history.values()):
                            logger.info("Real latency data collected, proceeding with initialization")
                            break
                        time.sleep(1)
                        wait_time += 1
                        logger.info(f"Waiting for real data... {wait_time}s")
                    
                    # Dump raw latency data structure after running checks
                    self.dump_latency_data()
                    
                    # Normalize the latency data format for compatibility with our display code
                    self.normalize_latency_data()
                    
                    # Update latency graphs with initial data
                    self.update_latency_graphs()
                    
                    # Update UI with real data
                    self.update_ui(results)
                else:
                    logger.warning("Warning: service_checker doesn't have run_service_checks method")
                    results = {}
            except Exception as e:
                logger.error(f"Error running service checks: {str(e)}")
                results = {}
                
        except Exception as e:
            logger.error(f"Critical error initializing data: {str(e)}")
            # Only add test data in case of critical error
            self.add_test_data()
            
    def normalize_latency_data(self):
        """Normalize the latency data format to be compatible with our display code"""
        if not hasattr(self.service_checker, 'latency_history'):
            logger.warning("service_checker has no latency_history attribute to normalize")
            return
            
        logger.info("Normalizing latency data format...")
        
        try:
            # Create a copy of the original data structure
            original_data = self.service_checker.latency_history.copy()
            normalized_data = {}
            
            # Map to translate endpoint formats
            domain_map = {}
            
            # Process each key in the original data
            for key, value in original_data.items():
                # Check if key is in "domain:port" format
                if isinstance(key, str) and ':' in key:
                    try:
                        # Extract domain and port
                        domain, port = key.split(':')
                        protocol = 'HTTPS' if port == '443' else 'HTTP' if port == '80' else f"PORT_{port}"
                        
                        # Store the mapping
                        domain_map[key] = domain
                        
                        # Create or update entry for this domain
                        if domain not in normalized_data:
                            normalized_data[domain] = {}
                            
                        # Add the data under the protocol key
                        protocol_key = f"{protocol}_{port}"
                        normalized_data[domain][protocol_key] = value
                        logger.info(f"Normalized {key} -> {domain} / {protocol_key}")
                    except Exception as e:
                        logger.warning(f"Error normalizing key {key}: {e}")
                        # Keep the original key
                        normalized_data[key] = value
                else:
                    # Not in domain:port format, keep as is
                    normalized_data[key] = value
            
            # Update the service_checker with the normalized data
            self.service_checker.latency_history = normalized_data
            
            # Store the domain map for future reference
            self.service_checker.domain_map = domain_map
            
            logger.info(f"Latency data normalized. New keys: {list(normalized_data.keys())}")
        except Exception as e:
            logger.error(f"Error normalizing latency data: {e}", exc_info=True)
            
    def dump_latency_data(self):
        """Dump latency data structure to logs for debugging"""
        logger.info("===== LATENCY DATA STRUCTURE DUMP =====")
        
        if not hasattr(self.service_checker, 'latency_history'):
            logger.warning("Service checker has no latency_history attribute")
            return
            
        if not self.service_checker.latency_history:
            logger.warning("Service checker latency_history is empty")
            return
            
        # First level: dump the structure summary
        logger.info(f"Top-level keys in latency_history: {list(self.service_checker.latency_history.keys())}")
        
        # Second level: check each data type
        for key, value in self.service_checker.latency_history.items():
            logger.info(f"Key: {key} (Type: {type(key).__name__})")
            
            if isinstance(value, dict):
                logger.info(f"  Contains {len(value)} nested keys: {list(value.keys())}")
                
                # Sample a few nested values
                for nested_key, nested_value in list(value.items())[:3]:  # Sample first 3
                    logger.info(f"  Nested key: {nested_key} (Type: {type(nested_key).__name__})")
                    
                    # If it's a deque or list or similar collection
                    if hasattr(nested_value, '__len__'):
                        logger.info(f"    Contains {len(nested_value)} data points (Type: {type(nested_value).__name__})")
                        
                        # Sample data points to see structure
                        if len(nested_value) > 0:
                            try:
                                sample = list(nested_value)[0] if hasattr(nested_value, '__iter__') else nested_value[0]
                                logger.info(f"    Sample data point: {sample} (Type: {type(sample).__name__})")
                                
                                # For tuples, check if it's (timestamp, value) format
                                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                                    logger.info(f"    Appears to be (timestamp, value) format: {sample[0]} -> {sample[1]}")
                            except (IndexError, TypeError) as e:
                                logger.warning(f"    Error sampling data: {e}")
                    else:
                        logger.info(f"    Not a collection: {nested_value}")
            elif hasattr(value, '__len__') and not isinstance(value, str):
                # It's a collection but not a dict
                logger.info(f"  Contains {len(value)} items (Type: {type(value).__name__})")
                
                # Sample data
                if len(value) > 0:
                    try:
                        sample = list(value)[0] if hasattr(value, '__iter__') else value[0]
                        logger.info(f"  Sample data point: {sample} (Type: {type(sample).__name__})")
                    except (IndexError, TypeError) as e:
                        logger.warning(f"  Error sampling data: {e}")
            else:
                # It's a direct value
                logger.info(f"  Direct value: {value} (Type: {type(value).__name__})")
        
        # Log actual latency data samples for key endpoints
        key_endpoints = [
            'teams.microsoft.com',
            'outlook.office365.com',
            'login.microsoftonline.com',
            'teams.microsoft.com:443',
            'outlook.office365.com:443'
        ]
        
        logger.info("===== LATENCY DATA SAMPLES FOR KEY ENDPOINTS =====")
        
        for endpoint in key_endpoints:
            # Check if the endpoint exists directly
            if endpoint in self.service_checker.latency_history:
                data = self.service_checker.latency_history[endpoint]
                logger.info(f"Found data for {endpoint} directly")
                if hasattr(data, '__len__'):
                    logger.info(f"  Contains {len(data)} data points")
                    if len(data) > 0:
                        try:
                            samples = list(data)[:3] if hasattr(data, '__iter__') else data[:3]
                            logger.info(f"  Samples: {samples}")
                        except (IndexError, TypeError) as e:
                            logger.warning(f"  Error sampling data: {e}")
                continue
                
            # Check service names
            found = False
            for service_name in self.service_checker.latency_history:
                if isinstance(self.service_checker.latency_history[service_name], dict):
                    if endpoint in self.service_checker.latency_history[service_name]:
                        data = self.service_checker.latency_history[service_name][endpoint]
                        logger.info(f"Found data for {endpoint} under service {service_name}")
                        if hasattr(data, '__len__'):
                            logger.info(f"  Contains {len(data)} data points")
                            if len(data) > 0:
                                try:
                                    samples = list(data)[:3] if hasattr(data, '__iter__') else data[:3]
                                    logger.info(f"  Samples: {samples}")
                                except (IndexError, TypeError) as e:
                                    logger.warning(f"  Error sampling data: {e}")
                        found = True
                        break
            
            if not found:
                logger.info(f"No data found for {endpoint}")
        
        # Test finding data with the LatencyGraph's method
        logger.info("===== TESTING LATENCY DATA ACCESS WITH LatencyGraph =====")
        test_endpoints = [
            'teams.microsoft.com',
            'outlook.office365.com',
            'login.microsoftonline.com'
        ]
        
        for endpoint in test_endpoints:
            # Create a temporary graph object to test data access
            temp_graph = LatencyGraph(None, self.service_checker, endpoint, auto_generate_test_data=False)
            latency_data = temp_graph.find_latency_data()
            
            if latency_data:
                logger.info(f"LatencyGraph.find_latency_data successfully found data for {endpoint}")
                if hasattr(latency_data, '__len__'):
                    logger.info(f"  Contains {len(latency_data)} data points")
                    if len(latency_data) > 0:
                        try:
                            samples = list(latency_data)[:3] if hasattr(latency_data, '__iter__') else latency_data[:3]
                            logger.info(f"  Samples: {samples}")
                        except (IndexError, TypeError) as e:
                            logger.warning(f"  Error sampling data: {e}")
            else:
                logger.warning(f"LatencyGraph.find_latency_data could not find data for {endpoint}")
        
        logger.info("===== END OF LATENCY DATA DUMP =====")
    
    def add_test_data(self):
        """Add test data for development purposes"""
        try:
            logger.info("Adding test data for development...")
            
            # Initialize latency_history if it doesn't exist
            if not hasattr(self.service_checker, 'latency_history'):
                logger.info("Creating new latency_history on service_checker")
                self.service_checker.latency_history = {}
                
            # Define domain-to-service mapping
            domain_service_map = {
                'teams.microsoft.com': 'Microsoft Teams',
                'presence.teams.microsoft.com': 'Microsoft Teams',
                'outlook.office365.com': 'Exchange Online',
                'sharepoint.com': 'SharePoint & OneDrive',
                'graph.microsoft.com': 'Microsoft Graph',
                'login.microsoftonline.com': 'Microsoft Authentication',
            }
                
            # Add some test latency points for all core endpoints
            current_time = datetime.now()
            
            # Make sure core_endpoints is defined
            if not hasattr(self, 'core_endpoints'):
                self.core_endpoints = [
                    'teams.microsoft.com',
                    'presence.teams.microsoft.com',
                    'outlook.office365.com',
                    'sharepoint.com',
                    'graph.microsoft.com',
                    'login.microsoftonline.com'
                ]
            
            # Create test data for each endpoint
            for endpoint in self.core_endpoints:
                logger.info(f"Creating test data for {endpoint}")
                # Create or update the endpoint in latency_history
                if endpoint not in self.service_checker.latency_history:
                    self.service_checker.latency_history[endpoint] = {}
                
                # Generate test data points for the last 15 minutes
                test_latency = []
                for i in range(60):  # 15 minutes (60 points at 15s intervals)
                    point_time = current_time - timedelta(seconds=15*i)
                    # Generate random latency between 40-80ms with some spikes
                    latency = 50 + 30 * (0.5 - 0.5 * (i % 10 == 0))  # Spike every 10th point
                    test_latency.append((point_time, latency))
                
                # Add latest point with high latency to test alerts
                test_latency.insert(0, (current_time, 150))
                
                # Create a collections.deque for the test data
                self.service_checker.latency_history[endpoint]['HTTPS_443'] = collections.deque(test_latency, maxlen=240)
                
                logger.info(f"Added {len(test_latency)} test data points for {endpoint}")
                
                # Log the structure of the latency_history
                logger.debug(f"latency_history structure: {self.service_checker.latency_history.keys()}")
                for domain in self.service_checker.latency_history:
                    logger.debug(f"  {domain} protocols: {self.service_checker.latency_history[domain].keys()}")
            
            # Add domain mappings to ServiceChecker 
            # (needed because the checker stores data by domain but looks up by endpoint)
            if not hasattr(self.service_checker, 'domain_service_map'):
                self.service_checker.domain_service_map = domain_service_map
                
            # Add some additional methods to service_checker if they don't exist
            if not hasattr(self.service_checker, 'get_baseline_range'):
                logger.info("Adding get_baseline_range method to service_checker")
                def get_baseline_range(endpoint):
                    """Simple stub for get_baseline_range"""
                    logger.debug(f"get_baseline_range called for {endpoint}")
                    history = self.service_checker.latency_history.get(endpoint, {}).get('HTTPS_443', [])
                    if not history:
                        logger.debug(f"No history found for {endpoint}")
                        return 0, 0
                    latencies = [l for _, l in history]
                    if not latencies:
                        logger.debug(f"No latencies found for {endpoint}")
                        return 0, 0
                    baseline_min = min(latencies) * 0.9
                    baseline_max = max(latencies) * 1.1
                    logger.debug(f"Baseline range for {endpoint}: {baseline_min:.2f}-{baseline_max:.2f}")
                    return baseline_min, baseline_max
                
                self.service_checker.get_baseline_range = get_baseline_range
            
            if not hasattr(self.service_checker, 'get_baseline_stability'):
                logger.info("Adding get_baseline_stability method to service_checker")
                def get_baseline_stability(endpoint):
                    """Simple stub for get_baseline_stability"""
                    logger.debug(f"get_baseline_stability called for {endpoint}")
                    baseline_min, baseline_max = self.service_checker.get_baseline_range(endpoint)
                    range_diff = baseline_max - baseline_min
                    if range_diff > 120:
                        return 'red'
                    elif range_diff > 60:
                        return 'orange'
                    return 'green'
                
                self.service_checker.get_baseline_stability = get_baseline_stability
            
            if not hasattr(self.service_checker, 'has_alert'):
                logger.info("Adding has_alert method to service_checker")
                def has_alert(endpoint):
                    """Simple stub for has_alert"""
                    # 20% chance of alert for test data
                    import random
                    result = random.random() < 0.2
                    logger.debug(f"has_alert called for {endpoint}, result: {result}")
                    return result
                
                self.service_checker.has_alert = has_alert
            
            if not hasattr(self.service_checker, 'stability_thresholds'):
                logger.info("Adding stability_thresholds to service_checker")
                self.service_checker.stability_thresholds = {
                    'green': 60,
                    'orange': 120,
                    'red': float('inf')
                }
            
            # Create test result data for tables
            test_results = self.create_test_results()
            
            # Store these results in the service checker for consistency
            if hasattr(self.service_checker, 'last_results'):
                self.service_checker.last_results = test_results
            
            # Dump the test results structure to the log
            logger.debug(f"Test results structure: {list(test_results.keys())}")
            for service in test_results:
                logger.debug(f"  Service {service} has {len(test_results[service].get('endpoints', []))} endpoints")
                
            # Update UI with test data
            self.update_ui(test_results)
            
            logger.info("Test data added successfully")
        except Exception as e:
            logger.error(f"Error adding test data: {str(e)}", exc_info=True)
            print(f"Error adding test data: {str(e)}")
    
    def create_test_results(self):
        """Create test results data structure for development"""
        # Create a data structure that exactly matches what service_checker.run_service_checks() returns
        # This should match the format expected by update_detailed_table
        
        service_map = {
            'teams.microsoft.com': 'Microsoft Teams',
            'presence.teams.microsoft.com': 'Microsoft Teams',
            'outlook.office365.com': 'Exchange Online',
            'sharepoint.com': 'SharePoint & OneDrive',
            'graph.microsoft.com': 'Microsoft Graph',
            'login.microsoftonline.com': 'Microsoft Authentication'
        }
        
        test_results = {}
        
        # Create entries for each service
        for domain, service_name in service_map.items():
            # Create service entry if it doesn't exist
            if service_name not in test_results:
                test_results[service_name] = {
                    'endpoints': []
                }
            
            # Create endpoint data
            endpoint_data = {
                'domain': domain,
                'http_check': True,
                'ports': []
            }
            
            # Add HTTPS port by default
            https_port = {
                'port': 443,
                'protocol': 'HTTPS',
                'is_healthy': True,
                'latency': 65.0,
                'error': ''
            }
            endpoint_data['ports'].append(https_port)
            
            # Add HTTP port for certain services
            if domain in ['teams.microsoft.com', 'sharepoint.com']:
                http_port = {
                    'port': 80,
                    'protocol': 'HTTP',
                    'is_healthy': True,
                    'latency': 60.0,
                    'error': ''
                }
                endpoint_data['ports'].append(http_port)
            
            # Add an unhealthy SMTP port to Exchange Online for testing
            if domain == 'outlook.office365.com':
                smtp_port = {
                    'port': 587,
                    'protocol': 'SMTP-TLS',
                    'is_healthy': False,
                    'latency': 0.0,
                    'error': 'Connection timed out'
                }
                endpoint_data['ports'].append(smtp_port)
            
            # Add endpoint to service
            test_results[service_name]['endpoints'].append(endpoint_data)
        
        return test_results 