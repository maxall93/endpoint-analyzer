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
import copy

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QSplitter, QTextEdit, QMessageBox, 
    QStatusBar, QSizePolicy, QScrollArea, QGroupBox, QGridLayout,
    QFrame, QSpacerItem, QComboBox, QDialog, QLineEdit, QCheckBox,
    QFormLayout
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

# Log management constants
MAX_LOG_LINES = 100000  # Maximum lines per log file
MAX_LOG_FOLDER_SIZE = 1024 * 1024 * 1024  # 1 GB in bytes
LOG_CHECK_INTERVAL = 3600  # Check log size every hour

def get_file_line_count(file_path):
    """Count the number of lines in a file"""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}")
        return 0

def get_folder_size(folder_path):
    """Calculate total size of files in a folder"""
    total_size = 0
    try:
        for path in Path(folder_path).rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
    except Exception as e:
        print(f"Error calculating folder size: {e}")
    return total_size

def cleanup_old_logs(log_dir, max_size):
    """Remove oldest log files until folder size is under max_size"""
    try:
        # Get all log files with their creation times
        log_files = [(f, f.stat().st_ctime) for f in Path(log_dir).glob('*.log')]
        log_files.sort(key=lambda x: x[1])  # Sort by creation time
        
        # Remove oldest files until under size limit
        current_size = get_folder_size(log_dir)
        while current_size > max_size and log_files:
            oldest_file = log_files.pop(0)[0]
            current_size -= oldest_file.stat().st_size
            oldest_file.unlink()
            print(f"Removed old log file: {oldest_file}")
            
    except Exception as e:
        print(f"Error cleaning up old logs: {e}")

def check_and_rotate_logs(current_log_file):
    """Check if current log file needs rotation and create new one if needed"""
    try:
        if not current_log_file.exists():
            return current_log_file
            
        line_count = get_file_line_count(current_log_file)
        if line_count >= MAX_LOG_LINES:
            # Create new log file
            new_log_file = Path('logs') / f'gui_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            print(f"Rotating log file to: {new_log_file}")
            return new_log_file
    except Exception as e:
        print(f"Error checking log rotation: {e}")
    
    return current_log_file

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'gui_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Clean up old logs before starting
cleanup_old_logs(log_dir, MAX_LOG_FOLDER_SIZE)

# Configure logging
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

# Create a timer for periodic log management
class LogManager:
    def __init__(self):
        self.current_log_file = log_file
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_logs)
        self.timer.start(LOG_CHECK_INTERVAL * 1000)  # Convert seconds to milliseconds
        
    def check_logs(self):
        """Periodic check of log files"""
        try:
            # Check if current log needs rotation
            new_log_file = check_and_rotate_logs(self.current_log_file)
            if new_log_file != self.current_log_file:
                # Update logging to use new file
                for handler in logger.handlers[:]:
                    if isinstance(handler, logging.FileHandler):
                        handler.close()
                        logger.removeHandler(handler)
                logger.addHandler(logging.FileHandler(new_log_file))
                self.current_log_file = new_log_file
                logger.info(f"Switched to new log file: {new_log_file}")
            
            # Clean up old logs if folder is too large
            cleanup_old_logs(log_dir, MAX_LOG_FOLDER_SIZE)
            
        except Exception as e:
            print(f"Error in log management: {e}")
            
    def cleanup(self):
        """Final cleanup of logs"""
        cleanup_old_logs(log_dir, MAX_LOG_FOLDER_SIZE)
        self.timer.stop()

# Create log manager instance
log_manager = LogManager()

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
        # Use the service_checker's method if available
        if hasattr(self.service_checker, 'get_service_for_domain'):
            service = self.service_checker.get_service_for_domain(endpoint_name)
            if service:
                return service
                
        # Fallback to hardcoded mapping if service_checker method not available or returns None
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


class AddEndpointDialog(QDialog):
    """Dialog for adding a new service endpoint"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Endpoint")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Service Category
        self.service_combo = QComboBox()
        self.service_combo.addItems([
            "Microsoft Teams",
            "Exchange Online",
            "SharePoint & OneDrive",
            "Microsoft Graph",
            "Other Services"
        ])
        form_layout.addRow("Service Category:", self.service_combo)
        
        # Domain/URL
        self.domain_edit = QLineEdit()
        self.domain_edit.setPlaceholderText("e.g., teams.microsoft.com")
        form_layout.addRow("Domain/URL:", self.domain_edit)
        
        # Description
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("Brief description of the endpoint")
        form_layout.addRow("Description:", self.description_edit)
        
        # Protocol Configuration Group
        protocol_group = QGroupBox("Protocol Configuration")
        protocol_layout = QVBoxLayout()
        
        # Protocol checks
        self.dns_check = QCheckBox("DNS Check")
        self.dns_check.setChecked(True)
        protocol_layout.addWidget(self.dns_check)
        
        self.https_check = QCheckBox("HTTPS (Port 443)")
        self.https_check.setChecked(True)
        protocol_layout.addWidget(self.https_check)
        
        self.http_check = QCheckBox("HTTP (Port 80)")
        protocol_layout.addWidget(self.http_check)
        
        # Health Check Configuration
        health_check_label = QLabel("Health Check Type:")
        health_check_label.setStyleSheet("margin-top: 10px;")
        protocol_layout.addWidget(health_check_label)
        
        self.health_check_combo = QComboBox()
        self.health_check_combo.addItems([
            "TCP Port Only",
            "Full Protocol Check"
        ])
        protocol_layout.addWidget(self.health_check_combo)
        
        # Custom Port
        custom_port_layout = QHBoxLayout()
        self.custom_port_check = QCheckBox("Custom Port:")
        self.custom_port_edit = QLineEdit()
        self.custom_port_edit.setPlaceholderText("Port number")
        self.custom_port_protocol = QComboBox()
        self.custom_port_protocol.addItems(["TCP", "UDP", "SMTP", "SMTP-TLS", "IMAP", "POP3"])
        custom_port_layout.addWidget(self.custom_port_check)
        custom_port_layout.addWidget(self.custom_port_edit)
        custom_port_layout.addWidget(self.custom_port_protocol)
        protocol_layout.addLayout(custom_port_layout)
        
        protocol_group.setLayout(protocol_layout)
        layout.addLayout(form_layout)
        layout.addWidget(protocol_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
    def get_endpoint_data(self):
        """Get the endpoint data from the form"""
        ports = []
        checks = []
        
        # Add DNS check if enabled
        if self.dns_check.isChecked():
            checks.append({
                "type": "DNS",
                "enabled": True
            })
        
        # Add HTTPS port and check if enabled
        if self.https_check.isChecked():
            ports.append({
                "port": 443,
                "protocol": "HTTPS"
            })
            if self.health_check_combo.currentText() == "Full Protocol Check":
                checks.append({
                    "type": "HTTPS",
                    "enabled": True
                })
            
        # Add HTTP port and check if enabled
        if self.http_check.isChecked():
            ports.append({
                "port": 80,
                "protocol": "HTTP"
            })
            if self.health_check_combo.currentText() == "Full Protocol Check":
                checks.append({
                    "type": "HTTP",
                    "enabled": True
                })
            
        # Add custom port if checked and valid
        if self.custom_port_check.isChecked():
            try:
                port = int(self.custom_port_edit.text())
                if 1 <= port <= 65535:
                    protocol = self.custom_port_protocol.currentText()
                    ports.append({
                        "port": port,
                        "protocol": protocol
                    })
                    # Add protocol check if selected
                    if self.health_check_combo.currentText() == "Full Protocol Check":
                        checks.append({
                            "type": protocol,
                            "enabled": True
                        })
            except ValueError:
                pass
        
        return {
            "service": self.service_combo.currentText(),
            "endpoint": {
                "domain": self.domain_edit.text(),
                "description": self.description_edit.text(),
                "ports": ports,
                "checks": checks
            }
        }

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
        
        # Add refresh button and add endpoint button in top right corner
        header_layout = QHBoxLayout()
        header_layout.addStretch()
        
        # Time interval selection combo box
        time_interval_label = QLabel("Time Interval:")
        time_interval_label.setStyleSheet("color: #dddddd;")
        self.time_interval_combo = QComboBox()
        self.time_interval_combo.addItems(["5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"])
        self.time_interval_combo.setCurrentIndex(1)  # Default to 15 minutes
        self.time_interval_combo.currentIndexChanged.connect(self.update_time_interval)
        
        # Add Endpoint button
        add_endpoint_button = QPushButton("Add Endpoint")
        add_endpoint_button.clicked.connect(self.show_add_endpoint_dialog)
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_data)
        
        header_layout.addWidget(time_interval_label)
        header_layout.addWidget(self.time_interval_combo)
        header_layout.addWidget(add_endpoint_button)
        header_layout.addWidget(refresh_button)
        
        main_layout.insertLayout(0, header_layout)
        
        # Status bar
        self.statusBar().showMessage("Initializing...")
        
        # Setup data manager thread
        self.data_manager = DataManagerThread(self.service_checker)
        self.data_manager.data_ready.connect(self.update_ui)
        self.data_manager.latency_updated.connect(self.update_latency_graphs)
        self.data_manager.status_message.connect(self.statusBar().showMessage)
        self.data_manager.initialization_complete.connect(self.on_initialization_complete)
        
        # Start the data manager thread
        self.data_manager.start()
        
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
        self.detailed_table = QTableWidget(0, 7)
        self.detailed_table.setHorizontalHeaderLabels([
            'Status', 
            'Service', 
            'Endpoint', 
            'Protocol',
            'Port',
            'Check Type',
            'Details'
        ])
        
        # Make header interactive and movable
        header = self.detailed_table.horizontalHeader()
        header.setSectionsMovable(True)
        header.setStretchLastSection(True)  # Last section (Details) stretches
        
        # Set default column widths and make them resizable
        default_widths = {
            0: 60,    # Status
            1: 150,   # Service
            2: 200,   # Endpoint
            3: 80,    # Protocol
            4: 60,    # Port
            5: 150,   # Check Type
            6: -1     # Details (-1 means stretch)
        }
        
        for col, width in default_widths.items():
            if width > 0:
                self.detailed_table.setColumnWidth(col, width)
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        
        # Make the Details column stretch to fill remaining space
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        
        # Other table properties
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
            
            # Get endpoints from the service checker
            self.core_endpoints = self.get_core_endpoints()
            
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
            
    def get_core_endpoints(self):
        """Get the list of core endpoints from the service checker"""
        endpoints = []
        
        # Check if service_checker has endpoints
        if not hasattr(self.service_checker, 'endpoints') or not self.service_checker.endpoints:
            logger.warning("No endpoints defined in service_checker")
            return endpoints
            
        # Extract domains from each service category
        for service_name, service_data in self.service_checker.endpoints.items():
            for endpoint_data in service_data.get('endpoints', []):
                domain = endpoint_data.get('domain', '')
                if domain:
                    endpoints.append(domain)
                    
        logger.info(f"Found {len(endpoints)} core endpoints")
        return endpoints
        
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
        
        try:
            # Process results into row data
            for service_name, service_data in results.items():
                # Process DNS checks
                for dns_check in service_data.get('dns_checks', []):
                    domain = dns_check.get('endpoint', '')
                    result = dns_check.get('result', {})
                    is_healthy = result.get('success', False)
                    latency = result.get('response_time', 0)
                    error = result.get('error', '')
                    
                    check_info = self.get_check_type_info('DNS', {'DNS': True})
                    detail_msg = self.format_detail_message(None, latency, error, is_healthy)
                    
                    row_data = [
                        is_healthy,
                        service_name,
                        domain,
                        'DNS',
                        'N/A',
                        check_info,
                        detail_msg
                    ]
                    
                    if is_healthy:
                        healthy_rows.append(row_data)
                    else:
                        unhealthy_rows.append(row_data)
                
                # Process port checks
                for port_check in service_data.get('port_checks', []):
                    endpoint = port_check.get('endpoint', '')
                    result = port_check.get('result', {})
                    is_healthy = result.get('success', False)
                    latency = result.get('latency_ms', 0)
                    error = result.get('error', '')
                    
                    # Split endpoint into domain and port
                    domain, port = endpoint.rsplit(':', 1) if ':' in endpoint else (endpoint, '')
                    
                    # Determine protocol based on port
                    protocol = 'TCP'
                    if port == '443':
                        protocol = 'HTTPS'
                    elif port == '80':
                        protocol = 'HTTP'
                    elif port == '587':
                        protocol = 'SMTP-TLS'
                    elif port == '993':
                        protocol = 'IMAP'
                    elif port == '995':
                        protocol = 'POP3'
                    
                    check_info = self.get_check_type_info(protocol, {protocol: False})  # Port check only
                    detail_msg = self.format_detail_message(None, latency, error, is_healthy)
                    
                    row_data = [
                        is_healthy,
                        service_name,
                        domain,
                        protocol,
                        port,
                        check_info,
                        detail_msg
                    ]
                    
                    if is_healthy:
                        healthy_rows.append(row_data)
                    else:
                        unhealthy_rows.append(row_data)
                
                # Process HTTP checks
                for http_check in service_data.get('http_checks', []):
                    endpoint = http_check.get('endpoint', '')
                    result = http_check.get('result', {})
                    is_healthy = result.get('success', False)
                    latency = result.get('latency_ms', 0)
                    error = result.get('error', '')
                    status_code = result.get('status_code', '')
                    
                    # Extract protocol and domain
                    protocol = 'HTTPS' if endpoint.startswith('https://') else 'HTTP'
                    domain = endpoint.split('://')[-1].split('/')[0]
                    port = '443' if protocol == 'HTTPS' else '80'
                    
                    check_info = self.get_check_type_info(protocol, {protocol: True})  # Full protocol check
                    detail_msg = self.format_detail_message(status_code, latency, error, is_healthy)
                    
                    row_data = [
                        is_healthy,
                        service_name,
                        domain,
                        protocol,
                        port,
                        check_info,
                        detail_msg
                    ]
                    
                    if is_healthy:
                        healthy_rows.append(row_data)
                    else:
                        unhealthy_rows.append(row_data)
        
        except Exception as e:
            logger.error(f"Error updating detailed table: {str(e)}", exc_info=True)
        
        # Add all rows to table
        all_rows = unhealthy_rows + healthy_rows
        
        for row_data in all_rows:
            row_position = self.detailed_table.rowCount()
            self.detailed_table.insertRow(row_position)
            
            # Status icon
            status_label = QLabel()
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setStyleSheet(
                "background-color: #4CAF50; border-radius: 10px;" if row_data[0] 
                else "background-color: #F44336; border-radius: 10px;"
            )
            status_label.setText("✓" if row_data[0] else "✗")
            self.detailed_table.setCellWidget(row_position, 0, status_label)
            
            # Other columns
            for col in range(1, len(row_data)):
                if col == 5:  # Check Type column
                    check_type, icon, color, tooltip = row_data[col]
                    check_label = QLabel(f"{icon} {check_type}")
                    check_label.setStyleSheet(f"color: {color}; padding: 2px 6px;")
                    check_label.setToolTip(tooltip)
                    self.detailed_table.setCellWidget(row_position, col, check_label)
                else:
                    item = QTableWidgetItem(str(row_data[col]))
                    if col == 6:  # Details column
                        item.setToolTip(str(row_data[col]))
                    self.detailed_table.setItem(row_position, col, item)
        
        logger.info(f"Detailed table updated with {self.detailed_table.rowCount()} rows")

    def get_check_type_info(self, protocol, check_types):
        """Get the check type information tuple"""
        check_type = "TCP Port Only"
        check_icon = "🔌"
        check_color = "#607D8B"
        check_tooltip = "Basic TCP port connectivity check"
        
        if protocol in check_types and check_types[protocol]:
            if protocol == "DNS":
                check_type = "Full DNS Check"
                check_icon = "🔍"
                check_color = "#9C27B0"
                check_tooltip = "Complete DNS resolution check including record validation"
            elif protocol in ["HTTP", "HTTPS"]:
                check_type = f"Full {protocol} Check"
                check_icon = "🌐"
                check_color = "#2196F3"
                check_tooltip = f"Complete {protocol} request with response validation"
            elif protocol == "SMTP":
                check_type = "Full SMTP Check"
                check_icon = "📧"
                check_color = "#FF9800"
                check_tooltip = "Complete SMTP connection and protocol check"
        
        return (check_type, check_icon, check_color, check_tooltip)

    def format_detail_message(self, status_code, latency, error, is_healthy):
        """Format the detail message for the table"""
        detail_msg = []
        if status_code:
            detail_msg.append(f"Status: {status_code}")
        if latency > 0:
            detail_msg.append(f"Latency: {latency:.1f}ms")
        if error:
            detail_msg.append(f"Error: {error}")
        elif is_healthy:
            detail_msg.append("Check passed successfully")
        
        return " | ".join(detail_msg) or "No additional details"
    
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
        self.statusBar().showMessage("Refreshing data...")
        # The data manager thread will handle the actual refresh
        # and update the UI via signals
        
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Stop the data manager thread
            if hasattr(self, 'data_manager'):
                self.data_manager.stop()
            
            # Clean up logs
            if hasattr(self, 'log_manager'):
                self.log_manager.cleanup()
                logger.info("Log manager cleanup completed")
            
            logger.info("Application shutting down")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            traceback.print_exc()
            
        event.accept()

    def on_initialization_complete(self):
        """Handle completion of data initialization"""
        self.statusBar().showMessage("Ready")

    def show_add_endpoint_dialog(self):
        """Show dialog for adding a new endpoint and process the result"""
        dialog = AddEndpointDialog(self)
        if dialog.exec():
            try:
                # Get the endpoint data from the dialog
                endpoint_data = dialog.get_endpoint_data()
                
                # Load current endpoints
                with open('endpoints.json', 'r') as f:
                    endpoints = json.load(f)
                
                # Add new endpoint to the appropriate service
                service_name = endpoint_data['service']
                if service_name not in endpoints:
                    endpoints[service_name] = {'endpoints': []}
                
                endpoints[service_name]['endpoints'].append(endpoint_data['endpoint'])
                
                # Save updated endpoints
                with open('endpoints.json', 'w') as f:
                    json.dump(endpoints, f, indent=4)
                
                # Refresh the service checker's endpoints
                self.service_checker.load_endpoints()
                
                # Update the latency trends tab
                self.setup_latency_trends_tab()
                
                # Refresh data to include new endpoint
                self.refresh_data()
                
                self.statusBar().showMessage("New endpoint added successfully")
            except Exception as e:
                logger.error(f"Error adding new endpoint: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to add endpoint: {str(e)}")
                self.statusBar().showMessage("Failed to add endpoint")

class DataManagerThread(QThread):
    """Thread for managing all data processing and service checks"""
    data_ready = pyqtSignal(dict)  # Signal emitted when new data is ready
    latency_updated = pyqtSignal()  # Signal emitted when latency data is updated
    initialization_complete = pyqtSignal()  # Signal emitted when initial setup is done
    status_message = pyqtSignal(str)  # Signal for updating status bar
    
    def __init__(self, service_checker):
        super().__init__()
        self.service_checker = service_checker
        self.running = True
        self.check_interval = 15000  # 15 seconds between checks
        
    def run(self):
        """Main thread loop"""
        try:
            # Initial setup
            self.status_message.emit("Loading endpoints...")
            
            # Initialize service checker
            if not hasattr(self.service_checker, 'endpoints') or not self.service_checker.endpoints:
                self.status_message.emit("No endpoints defined")
            else:
                endpoint_count = sum(len(service_data.get('endpoints', [])) 
                                for service_data in self.service_checker.endpoints.values())
                self.status_message.emit(f"Found {endpoint_count} endpoints")
            
            # Run initial service checks
            self.status_message.emit("Running initial service checks...")
            results = self.service_checker.run_service_checks()
            
            # Initialize latency history
            if hasattr(self.service_checker, 'latency_history'):
                self.status_message.emit("Collecting initial latency data...")
                # Wait for some initial data
                wait_time = 0
                while wait_time < 20 and self.running:
                    has_data = False
                    for service_data in self.service_checker.latency_history.values():
                        if isinstance(service_data, dict) and any(len(endpoint_data) > 0 
                            for endpoint_data in service_data.values() if hasattr(endpoint_data, '__len__')):
                            has_data = True
                            break
                    
                    if has_data:
                        break
                        
                    self.msleep(1000)  # Sleep for 1 second
                    wait_time += 1
                    self.status_message.emit(f"Collecting data... {wait_time}s")
            
            # Normalize data format
            self.normalize_latency_data()
            
            # Signal that initialization is complete
            self.initialization_complete.emit()
            
            # Emit initial data
            self.data_ready.emit(results)
            self.latency_updated.emit()
            
            # Main loop
            while self.running:
                try:
                    # Run service checks
                    results = self.service_checker.run_service_checks()
                    self.data_ready.emit(results)
                    self.latency_updated.emit()
                    
                    # Update status
                    self.status_message.emit(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    logger.error(f"Error in data manager loop: {str(e)}")
                    self.status_message.emit("Error updating data")
                
                # Wait for next check
                self.msleep(self.check_interval)
                
        except Exception as e:
            logger.error(f"Error in data manager thread: {str(e)}")
            self.status_message.emit("Error initializing data")
    
    def normalize_latency_data(self):
        """Normalize latency data format"""
        if not hasattr(self.service_checker, 'latency_history'):
            return
            
        try:
            original_structure = copy.deepcopy(self.service_checker.latency_history)
            
            for service_name, endpoints in original_structure.items():
                if not isinstance(endpoints, dict):
                    self.service_checker.latency_history[service_name] = {}
                    continue
                    
                for endpoint_key, data in endpoints.items():
                    if not data or not hasattr(data, '__iter__'):
                        continue
                    
                    try:
                        data_list = list(data)
                        if not data_list:
                            continue
                        
                        if isinstance(data, collections.deque) and all(
                            isinstance(item, tuple) and len(item) >= 2 and 
                            isinstance(item[0], datetime) and isinstance(item[1], (int, float))
                            for item in data_list
                        ):
                            continue
                        
                        normalized_data = collections.deque(maxlen=240)
                        now = datetime.now()
                        
                        if all(isinstance(item, (int, float)) for item in data_list):
                            for i, value in enumerate(reversed(data_list)):
                                timestamp = now - timedelta(seconds=i * 15)
                                normalized_data.appendleft((timestamp, float(value)))
                        elif all(isinstance(item, tuple) and len(item) >= 2 for item in data_list):
                            for item in data_list:
                                timestamp = item[0] if isinstance(item[0], datetime) else now
                                value = float(item[1]) if isinstance(item[1], (int, float)) else 0.0
                                normalized_data.append((timestamp, value))
                        
                        self.service_checker.latency_history[service_name][endpoint_key] = normalized_data
                        
                    except Exception as e:
                        logger.error(f"Error normalizing data for {endpoint_key}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error in normalize_latency_data: {str(e)}")
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()