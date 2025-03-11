import requests
import dns.resolver
import socket
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import collections
import copy
import time
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class ServiceChecker:
    def __init__(self, endpoints_file="endpoints.json"):
        # Load endpoints from JSON file
        self.endpoints = self.load_endpoints(endpoints_file)
        
        self.results_history = []
        self.max_history = 3600  # Store up to 1 hour of results (at 15s intervals)
        self.last_results = {}
        
        # Initialize latency history tracking
        self.latency_history = {}
        self._initialize_latency_history()
        
        # Baseline latency tracking - changed to rolling window
        self.baseline_window_minutes = 15
        
        # Alert tracking
        self.alert_data = {}
        self.alert_threshold_relative = (3, 5)  # Alert if 3 out of 5 values are above baseline max
        self.alert_threshold_absolute = 250  # Alert if latency is above 250ms
        self.alert_window_size = 5  # Number of data points to consider for alerting
        
        # Domain to service mapping for lookups
        self.domain_service_map = {}
        self.build_domain_service_map()
        
        # Stability thresholds for range difference
        self.stability_thresholds = {
            'green': 60,    # <60ms difference is green (stable)
            'orange': 120,  # 60-120ms difference is orange (stable but variable)
            'red': float('inf')  # >120ms difference is red (unstable)
        }

    def load_endpoints(self, endpoints_file):
        """Load endpoints from JSON file"""
        try:
            # Check if file exists
            if not os.path.exists(endpoints_file):
                logger.error(f"Endpoints file not found: {endpoints_file}")
                return {}
                
            # Load JSON file
            with open(endpoints_file, 'r') as f:
                data = json.load(f)
                
            logger.info(f"Loaded endpoints from {endpoints_file}")
            logger.debug(f"Endpoint categories: {list(data.get('categories', {}).keys())}")
            
            # Return the categories section which contains our endpoint structure
            return data.get('categories', {})
            
        except Exception as e:
            logger.error(f"Error loading endpoints from {endpoints_file}: {str(e)}")
            return {}
            
    def build_domain_service_map(self):
        """Build a mapping of domains to services for quick lookups"""
        self.domain_service_map = {}
        
        for service_name, service_data in self.endpoints.items():
            for endpoint in service_data.get('endpoints', []):
                domain = endpoint.get('domain', '')
                if domain:
                    self.domain_service_map[domain] = service_name
                    
        logger.debug(f"Built domain service map with {len(self.domain_service_map)} entries")
        
    def get_service_for_domain(self, domain):
        """Get the service name for a domain"""
        # Direct lookup
        if domain in self.domain_service_map:
            return self.domain_service_map[domain]
            
        # Partial match
        for key, service in self.domain_service_map.items():
            if key in domain or domain in key:
                return service
                
        # Try to infer service based on domain parts
        if 'teams' in domain:
            return 'Microsoft Teams'
        elif 'outlook' in domain or 'exchange' in domain or 'office365' in domain:
            return 'Exchange Online'
        elif 'sharepoint' in domain or 'onedrive' in domain:
            return 'SharePoint & OneDrive'
        elif 'graph' in domain:
            return 'Microsoft Graph'
        elif 'login' in domain or 'microsoftonline' in domain:
            return 'Microsoft Graph'  # Login endpoints are categorized under Graph
            
        # Default to unknown
        return None

    def check_dns_resolution(self, domain: str) -> Dict[str, Any]:
        """Test DNS resolution for a domain"""
        try:
            # Remove wildcard for DNS lookup
            test_domain = domain.replace('*.', 'www.')
            answers = dns.resolver.resolve(test_domain, 'A')
            return {
                'success': True,
                'ips': [str(rdata) for rdata in answers],
                'response_time': answers.response.time * 1000  # Convert to ms
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': None
            }

    def check_port(self, domain: str, port: int, protocol: str) -> Dict[str, Any]:
        """Check if a port is open on a domain and get latency"""
        try:
            socket_type = socket.SOCK_STREAM  # TCP
            
            # Resolve domain to IP address
            start_time = time.time()
            try:
                # Use getaddrinfo to get IP addresses for the domain
                addr_info = socket.getaddrinfo(domain, port, socket.AF_UNSPEC, socket_type)
                ip_address = addr_info[0][4][0]
            except socket.gaierror:
                return {
                    'success': False,
                    'error': 'Failed to resolve domain',
                    'latency_ms': None
                }
                
            # Create socket
            sock = socket.socket(socket.AF_INET, socket_type)
            sock.settimeout(5)  # 5 second timeout
            
            # Connect to server
            try:
                sock.connect((ip_address, port))
                connect_time = time.time()
                latency_ms = (connect_time - start_time) * 1000
                
                # For HTTPS protocol, try SSL handshake
                if protocol in ['HTTPS', 'SMTP-TLS']:
                    try:
                        context = ssl.create_default_context()
                        with context.wrap_socket(sock, server_hostname=domain) as ssl_sock:
                            ssl_time = time.time()
                            ssl_latency = (ssl_time - connect_time) * 1000
                            
                            # Add handshake time to latency
                            latency_ms += ssl_latency
                    except ssl.SSLError as e:
                        return {
                            'success': False,
                            'error': f'SSL Error: {str(e)}',
                            'latency_ms': latency_ms
                        }
                
                sock.close()
                return {
                    'success': True,
                    'latency_ms': latency_ms,
                    'error': None
                }
            except (socket.timeout, ConnectionRefusedError) as e:
                return {
                    'success': False,
                    'error': f'Connection failed: {str(e)}',
                    'latency_ms': None
                }
            finally:
                sock.close()
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency_ms': None
            }

    def check_http_endpoint(self, url: str, protocol: str = 'HTTPS') -> Dict[str, Any]:
        """Check an HTTP endpoint for connectivity and response time"""
        try:
            # Add protocol if not already present
            if not url.startswith('http'):
                url = f"{protocol.lower()}://{url}"
                
            # Extract domain for result
            domain = url.split('//')[1].split('/')[0]
                
            start_time = time.time()
            
            # Use requests to check connectivity
            timeout = 5  # 5 second timeout
            response = requests.get(url, timeout=timeout, verify=True)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Check for successful status code
            if response.status_code == 200:
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'latency_ms': latency_ms,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'latency_ms': latency_ms,
                    'error': f"Unexpected status code: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'status_code': None,
                'latency_ms': None,
                'error': 'Connection timed out'
            }
        except requests.exceptions.SSLError:
            return {
                'success': False,
                'status_code': None,
                'latency_ms': None,
                'error': 'SSL certificate verification failed'
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'status_code': None,
                'latency_ms': None,
                'error': 'Connection error'
            }
        except Exception as e:
            return {
                'success': False,
                'status_code': None,
                'latency_ms': None,
                'error': str(e)
            }

    def run_service_checks(self) -> Dict[str, Any]:
        """Run all service checks and return results"""
        timestamp = datetime.now()
        results = {}
        
        # Check if we have endpoints loaded
        if not self.endpoints:
            logger.error("No endpoints loaded. Cannot run service checks.")
            return {}
            
        logger.info(f"Running service checks for {len(self.endpoints)} services")
        
        # Process each service category
        for service_name, service_data in self.endpoints.items():
            logger.info(f"Checking service: {service_name}")
            
            # Initialize results for this service
            results[service_name] = {
                'dns_checks': [],
                'port_checks': [],
                'http_checks': []
            }
            
            # Process each endpoint in this service
            for endpoint_data in service_data.get('endpoints', []):
                domain = endpoint_data.get('domain', '')
                if not domain:
                    logger.warning(f"Skipping endpoint with no domain in {service_name}")
                    continue
                    
                logger.info(f"Checking endpoint: {domain}")
                
                # DNS resolution check
                dns_result = self.check_dns_resolution(domain)
                results[service_name]['dns_checks'].append({
                    'endpoint': domain,
                    'result': dns_result
                })
                
                # Skip further checks if DNS resolution failed
                if not dns_result.get('success', False):
                    logger.warning(f"DNS resolution failed for {domain}, skipping further checks")
                    continue
                
                # Port checks
                for port_info in endpoint_data.get('ports', []):
                    port = port_info.get('port', 0)
                    protocol = port_info.get('protocol', '')
                    
                    if not port or not protocol:
                        logger.warning(f"Skipping port check with incomplete data: {port_info}")
                        continue
                        
                    logger.info(f"Checking {protocol} port {port} on {domain}")
                    
                    # Run port check
                    port_result = self.check_port(domain, port, protocol)
                    endpoint_port = f"{domain}:{port}"
                    results[service_name]['port_checks'].append({
                        'endpoint': endpoint_port,
                        'result': port_result
                    })
                    
                    # Store latency data for trending if successful
                    if port_result.get('success', False) and 'latency_ms' in port_result:
                        latency = port_result['latency_ms']
                        logger.debug(f"Recording latency for {endpoint_port}: {latency}ms")
                        
                        # Ensure the service and endpoint exist in latency_history
                        if service_name not in self.latency_history:
                            self.latency_history[service_name] = {}
                        if endpoint_port not in self.latency_history[service_name]:
                            self.latency_history[service_name][endpoint_port] = collections.deque(maxlen=240)
                        
                        # Add the latency data point
                        self.latency_history[service_name][endpoint_port].append(
                            (timestamp, float(latency))
                        )
                        
                        # Update alert status
                        self.update_alert_status(endpoint_port, latency)
                
                # HTTP endpoint check if specified
                if endpoint_data.get('http_check', False):
                    for port_info in endpoint_data.get('ports', []):
                        protocol = port_info.get('protocol', '')
                        
                        # Only check HTTP/HTTPS protocols
                        if protocol not in ['HTTP', 'HTTPS']:
                            continue
                            
                        url = f"{protocol.lower()}://{domain}"
                        logger.info(f"Checking HTTP endpoint: {url}")
                        
                        http_result = self.check_http_endpoint(url, protocol)
                        results[service_name]['http_checks'].append({
                            'endpoint': url,
                            'result': http_result
                        })
                        
                        # Store latency data for trending if successful
                        if http_result.get('success', False) and 'latency_ms' in http_result:
                            latency = http_result['latency_ms']
                            logger.debug(f"Recording HTTP latency for {url}: {latency}ms")
                            http_endpoint = f"{domain}:{protocol}"
                            
                            # Ensure the service and endpoint exist in latency_history
                            if service_name not in self.latency_history:
                                self.latency_history[service_name] = {}
                            if http_endpoint not in self.latency_history[service_name]:
                                self.latency_history[service_name][http_endpoint] = collections.deque(maxlen=240)
                            
                            # Add the latency data point
                            self.latency_history[service_name][http_endpoint].append(
                                (timestamp, float(latency))
                            )
                            
                            # Update alert status
                            self.update_alert_status(http_endpoint, latency)
        
        # Save results for history
        self._save_results(results)
        
        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save results to history and update baselines"""
        # Save to history
        timestamp = datetime.now()
        self.results_history.append({
            "timestamp": timestamp,
            "results": copy.deepcopy(results)
        })
        
        # Trim history if needed
        if len(self.results_history) > self.max_history:
            self.results_history = self.results_history[-self.max_history:]
            
        # Save last results for reference
        self.last_results = results
    
    def get_rolling_baseline(self, endpoint: str) -> Dict[str, Any]:
        """Calculate the rolling baseline for an endpoint using the last 15 minutes of data"""
        # Get the latency history for this endpoint
        service = None
        data_points = []
        
        # Find the service this endpoint belongs to
        for svc, endpoints in self.latency_history.items():
            if endpoint in endpoints:
                service = svc
                data_points = list(endpoints[endpoint])
                break
        
        if not service or not data_points:
            return {
                "min": 0,
                "max": 0,
                "avg": 0,
                "avg_min": 0,
                "avg_max": 0,
                "range_diff": 0,
                "stability": "green",
                "has_data": False,
                "consecutive_alerts": 0
            }
            
        # Get data points from the last 15 minutes
        cutoff_time = datetime.now() - timedelta(minutes=self.baseline_window_minutes)
        recent_points = [(ts, latency) for ts, latency in data_points if ts >= cutoff_time]
        
        if not recent_points:
            return {
                "min": 0,
                "max": 0,
                "avg": 0,
                "avg_min": 0,
                "avg_max": 0,
                "range_diff": 0,
                "stability": "green",
                "has_data": False,
                "consecutive_alerts": 0
            }
            
        # Calculate absolute min, max, and average latency
        latencies = [latency for _, latency in recent_points]
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)
        
        # Calculate average min/max by splitting the data into segments
        segment_mins = []
        segment_maxs = []
        
        # Sort data points by timestamp
        recent_points.sort(key=lambda x: x[0])
        
        # Split into 5 segments (approximately 3 minutes each)
        num_segments = 5
        segment_size = max(1, len(recent_points) // num_segments)
        
        for i in range(0, len(recent_points), segment_size):
            segment = recent_points[i:i+segment_size]
            if segment:
                segment_latencies = [lat for _, lat in segment]
                segment_mins.append(min(segment_latencies))
                segment_maxs.append(max(segment_latencies))
        
        # Calculate average min and max
        avg_min = sum(segment_mins) / len(segment_mins) if segment_mins else min_latency
        avg_max = sum(segment_maxs) / len(segment_maxs) if segment_maxs else max_latency
        
        # Ensure avg_min and avg_max are different by at least 5ms
        if abs(avg_max - avg_min) < 5:
            mid = (avg_max + avg_min) / 2
            avg_min = mid - 2.5
            avg_max = mid + 2.5
            
        # Calculate range difference based on average min/max
        range_diff = avg_max - avg_min
        
        # Determine stability based on range difference
        stability = "green"
        if range_diff > self.stability_thresholds['orange']:
            stability = "red"
        elif range_diff > self.stability_thresholds['green']:
            stability = "orange"
            
        # Get number of consecutive alerts
        consecutive_alerts = self.alert_data.get(endpoint, {}).get("consecutive_alerts", 0)
        
        return {
            "min": min_latency,
            "max": max_latency,
            "avg": avg_latency,
            "avg_min": avg_min,
            "avg_max": avg_max,
            "range_diff": range_diff,
            "stability": stability,
            "has_data": True,
            "consecutive_alerts": consecutive_alerts
        }
        
    def update_alert_status(self, endpoint: str, current_latency: float) -> bool:
        """Update and return the alert status for an endpoint using both relative and absolute thresholds"""
        # Initialize alert data structure for this endpoint if needed
        if endpoint not in self.alert_data:
            self.alert_data[endpoint] = {
                "last_values": collections.deque(maxlen=10),  # Store last 10 values
                "alert_status": False,  # Current alert status
                "above_count": 0,       # Count of values above average max
                "high_latency_count": 0  # Count of values above absolute threshold
            }
        
        # Add current latency to history
        self.alert_data[endpoint]["last_values"].append(current_latency)
        
        # Get baseline data
        baseline = self.get_rolling_baseline(endpoint)
        if not baseline["has_data"]:
            return False
        
        # Check if current value exceeds absolute threshold
        is_high_latency = current_latency > self.alert_threshold_absolute
        
        # Check if current value exceeds average max
        is_above_avg_max = current_latency > baseline["avg_max"]
        
        # Calculate how many of the last values are above average max
        above_count = sum(1 for val in self.alert_data[endpoint]["last_values"] 
                         if val > baseline["avg_max"])
        
        # Calculate how many of the last values are high latency
        high_latency_count = sum(1 for val in self.alert_data[endpoint]["last_values"] 
                               if val > self.alert_threshold_absolute)
        
        # Update counts in alert data
        self.alert_data[endpoint]["above_count"] = above_count
        self.alert_data[endpoint]["high_latency_count"] = high_latency_count
        
        # Determine if we should alert based on relative threshold (50% above baseline)
        relative_alert = above_count >= 3  # Alert if 3 or more values are above avg_max
        
        # Determine if we should alert based on absolute threshold
        absolute_alert = high_latency_count >= 2  # Alert if 2 or more values are above absolute threshold
        
        # Alert if either condition is met
        should_alert = relative_alert or absolute_alert
        
        # Update and return alert status
        self.alert_data[endpoint]["alert_status"] = should_alert
        
        # Print diagnostic info for alerts
        if should_alert and not self.alert_data[endpoint].get("was_alerting", False):
            if relative_alert:
                print(f"ALERT: {endpoint} has {above_count}/{len(self.alert_data[endpoint]['last_values'])} values above avg max")
            if absolute_alert:
                print(f"HIGH LATENCY ALERT: {endpoint} has {high_latency_count} values above {self.alert_threshold_absolute}ms")
            self.alert_data[endpoint]["was_alerting"] = True
        elif not should_alert and self.alert_data[endpoint].get("was_alerting", False):
            print(f"Alert cleared for {endpoint}")
            self.alert_data[endpoint]["was_alerting"] = False
        
        return should_alert
        
    def has_alert(self, endpoint: str) -> bool:
        """Check if the endpoint currently has an active alert"""
        if endpoint not in self.alert_data:
            return False
        
        return self.alert_data[endpoint]["alert_status"]
    
    def get_alert_details(self, endpoint: str) -> Dict[str, Any]:
        """Get detailed alert information for an endpoint"""
        if endpoint not in self.alert_data:
            return {
                "alert_status": False,
                "above_count": 0,
                "high_latency_count": 0,
                "total_points": 0
            }
        
        alert_data = self.alert_data[endpoint]
        n_required, m_window = self.alert_threshold_relative
        
        details = {
            "alert_status": alert_data["alert_status"],
            "above_count": alert_data["above_count"],
            "high_latency_count": alert_data["high_latency_count"],
            "total_points": len(alert_data["last_values"]),
            "threshold_relative": f"{n_required} of {m_window}",
            "threshold_absolute": self.alert_threshold_absolute
        }
        
        return details

    def get_service_status(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Analyze results and return service status"""
        status = {}
        
        for service, checks in results.items():
            # Check if any DNS resolutions failed
            dns_failed = any(not check['result']['success'] 
                           for check in checks['dns_checks'])
            
            # Check if any port checks failed
            ports_failed = any(not check['result']['success'] 
                             for check in checks['port_checks'])
            
            # Check if any HTTP checks failed
            http_failed = any(not check['result']['success'] 
                            for check in checks['http_checks'])
            
            # Determine overall status
            if dns_failed and ports_failed:
                status[service] = 'Critical'
            elif dns_failed or ports_failed or http_failed:
                status[service] = 'Warning'
            else:
                status[service] = 'Healthy'
        
        return status 

    def get_latency_trends(self, hours: int = 1) -> Dict[str, Dict[str, List[Tuple[datetime, float]]]]:
        """Get latency trends for the specified number of hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        trends = {}
        
        for service, endpoints in self.latency_history.items():
            if service not in trends:
                trends[service] = {}
                
            for endpoint, data_points in endpoints.items():
                # Filter data points within the time range
                recent_points = [(ts, lat) for ts, lat in data_points if ts >= cutoff_time]
                
                # Only include endpoints with data
                if recent_points:
                    trends[service][endpoint] = recent_points
                    print(f"Found {len(recent_points)} data points for {service} - {endpoint}")
                
        # Debug output
        if not trends:
            print("No latency trend data available in the service checker.")
        else:
            print(f"Returning latency trends for {len(trends)} services.")
            
        return trends
        
    def calculate_trend_slope(self, data_points: List[Tuple[datetime, float]]) -> float:
        """Calculate the slope of the trend line using linear regression
        Positive value means increasing latency, negative means decreasing
        """
        if len(data_points) < 2:
            return 0.0
            
        # Convert timestamps to seconds since first data point for calculation
        base_time = data_points[0][0]
        x_values = [(ts - base_time).total_seconds() for ts, _ in data_points]
        y_values = [latency for _, latency in data_points]
        
        # Calculate means
        n = len(data_points)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n
        
        # Calculate slope using least squares method
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        denominator = sum((x - mean_x) ** 2 for x in x_values)
        
        # Avoid division by zero
        if denominator == 0:
            return 0.0
            
        return numerator / denominator 

    def is_latency_stable(self, endpoint: str, latency: float) -> bool:
        """Check if latency stability based on rolling baseline and thresholds"""
        baseline = self.get_rolling_baseline(endpoint)
        
        if not baseline["has_data"]:
            return True  # Consider stable if no baseline data yet
            
        # Only unstable if in the red zone (range > 120ms)
        return baseline["stability"] != "red"
    
    def get_baseline_range(self, endpoint: str) -> Tuple[float, float]:
        """Get the min and max baseline range for an endpoint"""
        baseline = self.get_rolling_baseline(endpoint)
        
        if not baseline["has_data"]:
            return (0, 0)  # Return zeros if no baseline data
            
        # Return average min/max instead of absolute min/max
        return (baseline["avg_min"], baseline["avg_max"])

    def get_baseline_stability(self, endpoint: str) -> str:
        """Get the stability color (green, orange, red) for an endpoint"""
        baseline = self.get_rolling_baseline(endpoint)
        
        if not baseline["has_data"]:
            return "green"  # Default to green if no data
            
        return baseline["stability"]
        
    def is_baseline_established(self, endpoint: str) -> bool:
        """Check if baseline has been established for this endpoint"""
        baseline = self.get_rolling_baseline(endpoint)
        return baseline["has_data"]

    def get_latency_stats(self, endpoint: str) -> Dict[str, Any]:
        """
        Calculate and return latency statistics for a given endpoint
        
        Args:
            endpoint: The endpoint name (can be domain or domain:port format)
            
        Returns:
            Dictionary with latency statistics:
            {
                'avg': float,  # Average latency in ms
                'min': float,  # Minimum latency in ms
                'max': float,  # Maximum latency in ms
                'current': float,  # Most recent latency in ms
                'samples': int,  # Number of samples used
                'has_data': bool  # Whether any data was found
            }
        """
        # Default return structure
        stats = {
            'avg': 0.0,
            'min': 0.0,
            'max': 0.0,
            'current': 0.0,
            'samples': 0,
            'has_data': False
        }
        
        # Find the latency data for this endpoint
        latency_data = None
        
        # First, try direct endpoint name
        for service, endpoints in self.latency_history.items():
            if endpoint in endpoints:
                latency_data = endpoints[endpoint]
                break
                
        # If not found, try to find endpoint:port format
        if not latency_data and ':' not in endpoint:
            for service, endpoints in self.latency_history.items():
                for ep in endpoints:
                    if ep.startswith(endpoint + ':'):
                        latency_data = endpoints[ep]
                        break
                if latency_data:
                    break
                    
        # If still not found, try substring match
        if not latency_data:
            for service, endpoints in self.latency_history.items():
                for ep in endpoints:
                    if endpoint in ep:
                        latency_data = endpoints[ep]
                        break
                if latency_data:
                    break
        
        # If we found data, calculate statistics
        if latency_data and len(latency_data) > 0:
            # Extract latency values from (timestamp, latency) tuples
            latencies = [point[1] for point in latency_data if isinstance(point, tuple) and len(point) >= 2]
            
            if latencies:
                stats['avg'] = sum(latencies) / len(latencies)
                stats['min'] = min(latencies)
                stats['max'] = max(latencies)
                stats['current'] = latencies[-1] if latencies else 0.0
                stats['samples'] = len(latencies)
                stats['has_data'] = True
        
        return stats

    def _initialize_latency_history(self):
        """Initialize the latency history structure for all services and endpoints"""
        if not self.endpoints:
            logger.warning("No endpoints loaded, latency history will be empty")
            return
            
        # Initialize the structure for each service and endpoint
        for service_name, service_data in self.endpoints.items():
            self.latency_history[service_name] = {}
            
            # Process each endpoint in this service
            for endpoint_data in service_data.get('endpoints', []):
                domain = endpoint_data.get('domain', '')
                if not domain:
                    logger.warning(f"Skipping endpoint with no domain in {service_name}")
                    continue
                
                # Initialize for each port
                for port_info in endpoint_data.get('ports', []):
                    port = port_info.get('port', 0)
                    if port:
                        endpoint_key = f"{domain}:{port}"
                        self.latency_history[service_name][endpoint_key] = collections.deque(maxlen=240)
                
                # Initialize for HTTP endpoints if specified
                if endpoint_data.get('http_check', False):
                    for port_info in endpoint_data.get('ports', []):
                        protocol = port_info.get('protocol', '')
                        if protocol in ['HTTP', 'HTTPS']:
                            http_endpoint = f"{domain}:{protocol}"
                            self.latency_history[service_name][http_endpoint] = collections.deque(maxlen=240)
        
        logger.info(f"Initialized latency history for {len(self.latency_history)} services")