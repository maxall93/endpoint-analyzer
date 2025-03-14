# M365 Endpoint Analyzer

######## CURSOR AGENTIC WORKFLOW POC #####
Created completely in Cursor using Claude Sonnet 3.5 and 3.7, not a single line of code has been manually inputted or edited. Test project to see the state-of-the-art in LLM based agentic coding workflows.
#########################################

A powerful desktop application for monitoring and analyzing Microsoft 365 service endpoints. The application provides real-time monitoring of endpoint health, latency tracking, and detailed status information with a modern, dark-themed user interface.

## Features

- Real-time monitoring of Microsoft 365 service endpoints
- Detailed status information for each endpoint including DNS, HTTP, and TCP checks
- Interactive latency graphs with historical data
- Service categorization (Teams, Exchange, SharePoint, etc.)
- Custom endpoint management
- Dark theme UI with responsive design
- Background processing for improved performance
- Comprehensive logging system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/365EndpointAnalyzer.git
cd 365EndpointAnalyzer
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python main.py
```

2. The application will open with three main tabs:
   - **Service Overview**: High-level status of each service category
   - **Detailed Status**: Comprehensive endpoint check results
   - **Latency Trends**: Real-time latency graphs for each endpoint

3. Use the top toolbar to:
   - Add new endpoints
   - Remove endpoints or specific ports
   - Restore from backups
   - Refresh data manually
   - Change the time interval for latency graphs

4. Monitor endpoint health through:
   - Color-coded status indicators
   - Detailed check information
   - Interactive latency graphs
   - Protocol-specific status details

## File Structure

- `main.py`: Application entry point and initialization
- `gui.py`: Main GUI implementation using PyQt6
  - Contains window layouts, widgets, and UI logic
  - Implements real-time data visualization
  - Handles user interactions and threading
- `service_checker.py`: Core service checking functionality
  - Implements endpoint health checks
  - Manages latency monitoring
  - Handles protocol-specific checks (DNS, HTTP, TCP)
- `endpoints.json`: Configuration file for monitored endpoints
  - Defines service categories
  - Specifies endpoint properties and check types
- `utils.py`: Utility functions and helper methods
- `requirements.txt`: Python package dependencies
- `logs/`: Directory containing application logs
  - Automatically rotated and managed
  - Includes detailed debugging information

## Configuration

### Adding New Endpoints

1. Click the "Add Endpoint" button
2. Select the service category
3. Enter the endpoint domain/URL
4. Configure check types:
   - DNS Check
   - HTTP/HTTPS Check
   - Custom Port Check
5. Choose between TCP Port Only or Full Protocol Check
6. Save the configuration

### Removing Endpoints

1. Click the "Remove Endpoint" button
2. Select a service category from the left panel
3. Select an endpoint from the category
4. Choose a removal option:
   - Remove specific ports (select ports from the list)
   - Remove the entire endpoint
   - Remove the entire category
5. Confirm the removal action
6. A backup of the current configuration will be automatically created

### Restoring from Backups

1. Click the "Restore Backup" button
2. Select a backup file from the list (sorted by date, newest first)
3. Confirm the restoration
4. The application will create a backup of the current configuration before restoring

### Endpoint Check Types

- **TCP Port Only**: Basic connectivity check
- **Full Protocol Check**: Complete protocol-specific validation
- **DNS Check**: Domain resolution and record validation
- **HTTP/HTTPS**: Complete web request validation

## Development

### Key Components

1. **GUI Layer** (`gui.py`):
   - PyQt6-based user interface
   - Real-time data visualization
   - Event handling and user interaction

2. **Service Layer** (`service_checker.py`):
   - Endpoint health monitoring
   - Protocol-specific checks
   - Latency tracking and history

3. **Data Management**:
   - Background processing
   - Thread-safe data handling
   - Efficient state management

### Threading Model

- Main UI thread for interface responsiveness
- Background thread for service checks
- Separate thread for data processing
- Signal-based communication between threads

## Logging

The application maintains detailed logs in the `logs/` directory:
- Automatic log rotation
- Size-based cleanup
- Comprehensive error tracking
- Performance monitoring

## Requirements

- Python 3.8 or higher
- PyQt6
- Matplotlib
- Additional dependencies listed in requirements.txt

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.