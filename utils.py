import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import dns.resolver
import requests
from requests.exceptions import RequestException

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Setup logging
def setup_logging():
    log_file = log_dir / f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for more detailed logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def log_api_response(endpoint: str, response: requests.Response):
    """Log API response details"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'status_code': response.status_code,
        'elapsed_time': response.elapsed.total_seconds()
    }
    
    logger.info(f"API Response: {json.dumps(log_entry)}")

def check_endpoint_connectivity(url: str, timeout: int = 5) -> Dict[str, Any]:
    """Check connectivity to an endpoint"""
    result = {
        'url': url,
        'timestamp': datetime.now().isoformat(),
        'dns_resolution': False,
        'http_connection': False,
        'response_time': None,
        'error': None
    }

    try:
        # DNS resolution check
        domain = url.split('://')[1].split('/')[0]
        dns.resolver.resolve(domain, 'A')
        result['dns_resolution'] = True

        # HTTP connection check
        start_time = datetime.now()
        response = requests.get(url, timeout=timeout)
        response_time = (datetime.now() - start_time).total_seconds()
        
        result.update({
            'http_connection': True,
            'response_time': response_time,
            'status_code': response.status_code
        })

    except dns.exception.DNSException as e:
        result['error'] = f"DNS resolution failed: {str(e)}"
        logger.error(f"DNS check failed for {url}: {str(e)}")

    except RequestException as e:
        result['error'] = f"HTTP connection failed: {str(e)}"
        logger.error(f"HTTP check failed for {url}: {str(e)}")

    return result

def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to human-readable format"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return timestamp

def save_results(results: Dict[str, Any], filename: str):
    """Save results to a JSON file"""
    file_path = log_dir / filename
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename: str) -> Optional[Dict[str, Any]]:
    """Load results from a JSON file"""
    file_path = log_dir / filename
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return None