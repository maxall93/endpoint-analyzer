# Microsoft 365 & Azure Service Status Dashboard

A Windows desktop application for monitoring Microsoft 365 and Azure service health and connectivity.

## Features

- Microsoft 365 Authentication using device code flow
- Real-time service health monitoring
- Connectivity testing for Microsoft services
- Interactive dashboard with status indicators
- Historical data logging
- Advanced latency trend analysis with visual graphs and alerts
  - Overview of HTTPS/443 connectivity for all services
  - Detailed service-specific latency monitoring
  - Flexible time window selection (5 min, 15 min, 30 min, 1 hour)
  - Smart rolling baseline detection (15-minute window)
    - Uses average min/max values from 5 segments (3 minutes each)
    - Values can exceed the average range, triggering alerts
    - Color-coded stability ranges:
      - Green: stable (<60ms variation)
      - Orange: variable but stable (60-120ms variation)
      - Red: unstable (>120ms variation)
    - Visual stability indicators showing current status
  - Sophisticated dual-trigger alert system:
    - Pattern detection: Alerts when 7 out of 10 recent values exceed the average range
    - Absolute threshold: Alerts when any values exceed 250ms latency  
    - Clear alert categories with specific messaging
    - Visual cues including background color changes and labeled alert types
    - Detailed diagnostic information in pop-up notifications
  - Absolute threshold line displayed on all graphs for reference

## Requirements

- Python 3.8 or higher
- Windows 10/11
- Microsoft 365 account with appropriate permissions

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. When prompted:
   - Visit the Microsoft device login page (URL will be displayed)
   - Enter the code shown in the application
   - Sign in with your Microsoft 365 account
   - Grant the requested permissions

After the initial authentication, the application will cache your credentials for future use.

## Development

The project structure is organized as follows:
- `main.py`: Application entry point
- `auth.py`: Microsoft authentication handling
- `service_checker.py`: Service health and connectivity checks
- `gui.py`: PyQt6-based user interface
- `config.py`: Configuration management
- `utils.py`: Helper utilities

## License

MIT License