#!/usr/bin/env python3
"""
Memory monitoring script for Resume Scorer.
Tracks memory usage and performs cleanup when needed.
Can be run as a background thread or standalone process.
Optimized for Render's free tier deployment.
"""

import os
import sys
import time
import gc
import psutil
import logging
import threading
import argparse
from datetime import datetime
import torch

# Add base directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Import config
try:
    from src.config import MEMORY_MONITORING_INTERVAL, ENABLE_MEMORY_MONITORING, ON_RENDER
except ImportError:
    MEMORY_MONITORING_INTERVAL = 60  # Default: check more frequently on Render
    ENABLE_MEMORY_MONITORING = True
    ON_RENDER = "RENDER" in os.environ

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(base_dir, "memory_monitor.log"))
    ]
)
logger = logging.getLogger("memory_monitor")

# Memory thresholds (percentage) - lower for Render free tier
WARNING_THRESHOLD = 70 if ON_RENDER else 80
CRITICAL_THRESHOLD = 80 if ON_RENDER else 90
EMERGENCY_THRESHOLD = 90 if ON_RENDER else 95

class MemoryMonitor:
    """Monitor memory usage and perform cleanup when needed"""
    
    def __init__(self, interval=MEMORY_MONITORING_INTERVAL):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.thread = None
        self.last_cleanup_time = 0
        self.cleanup_counter = 0
        
    def get_memory_usage(self):
        """Get current memory usage as percentage and absolute values"""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            system_percent = system_memory.percent
            system_used_mb = system_memory.used / (1024 * 1024)
            system_total_mb = system_memory.total / (1024 * 1024)
            
            # Process memory
            process_memory = self.process.memory_info()
            process_used_mb = process_memory.rss / (1024 * 1024)
            
            return {
                "system_percent": system_percent,
                "system_used_mb": system_used_mb,
                "system_total_mb": system_total_mb,
                "process_used_mb": process_used_mb,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return None
    
    def log_memory_usage(self):
        """Log current memory usage"""
        usage = self.get_memory_usage()
        if not usage:
            return
        
        # Only log every 5th check to reduce log size on Render
        self.cleanup_counter += 1
        if self.cleanup_counter % 5 == 0 or usage['system_percent'] > WARNING_THRESHOLD:
            logger.info(
                f"Memory usage: System {usage['system_percent']:.1f}% "
                f"({usage['system_used_mb']:.1f}MB / {usage['system_total_mb']:.1f}MB), "
                f"Process {usage['process_used_mb']:.1f}MB"
            )
        
        # Check thresholds and take action
        self._check_thresholds(usage)
    
    def _check_thresholds(self, usage):
        """Check memory thresholds and take action if needed"""
        system_percent = usage["system_percent"]
        process_mb = usage["process_used_mb"]
        
        # Only perform cleanup if enough time has passed since last cleanup
        current_time = time.time()
        time_since_cleanup = current_time - self.last_cleanup_time
        
        # On Render, be more aggressive with cleanups
        if ON_RENDER:
            if system_percent >= EMERGENCY_THRESHOLD and time_since_cleanup >= 30:
                logger.critical(f"EMERGENCY: Memory usage at {system_percent:.1f}%! Performing emergency cleanup")
                self._emergency_cleanup()
                self.last_cleanup_time = current_time
            elif system_percent >= CRITICAL_THRESHOLD and time_since_cleanup >= 60:
                logger.warning(f"CRITICAL: Memory usage at {system_percent:.1f}%! Performing aggressive cleanup")
                self._aggressive_cleanup()
                self.last_cleanup_time = current_time
            elif system_percent >= WARNING_THRESHOLD and time_since_cleanup >= 120:
                logger.warning(f"WARNING: Memory usage at {system_percent:.1f}%! Performing standard cleanup")
                self._standard_cleanup()
                self.last_cleanup_time = current_time
        else:
            # Standard thresholds for non-Render environments
            if system_percent >= EMERGENCY_THRESHOLD and time_since_cleanup >= 60:
                logger.critical(f"EMERGENCY: Memory usage at {system_percent:.1f}%! Performing emergency cleanup")
                self._emergency_cleanup()
                self.last_cleanup_time = current_time
            elif system_percent >= CRITICAL_THRESHOLD and time_since_cleanup >= 300:
                logger.warning(f"CRITICAL: Memory usage at {system_percent:.1f}%! Performing aggressive cleanup")
                self._aggressive_cleanup()
                self.last_cleanup_time = current_time
            elif system_percent >= WARNING_THRESHOLD and time_since_cleanup >= 600:
                logger.warning(f"WARNING: Memory usage at {system_percent:.1f}%! Performing standard cleanup")
                self._standard_cleanup()
                self.last_cleanup_time = current_time
    
    def _standard_cleanup(self):
        """Perform standard memory cleanup"""
        logger.info("Performing standard memory cleanup")
        
        # Run garbage collection (more aggressively)
        gc.collect(generation=2)
        
        # Clear PyTorch cache if available
        if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared PyTorch CUDA cache")
        
        # On Render, also trim memory
        if ON_RENDER:
            try:
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
                logger.info("Trimmed system memory with malloc_trim")
            except:
                logger.warning("Could not trim memory with malloc_trim")
    
    def _aggressive_cleanup(self):
        """Perform more aggressive memory cleanup"""
        logger.warning("Performing aggressive memory cleanup")
        
        # Standard cleanup first
        self._standard_cleanup()
        
        # More aggressive steps
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Reset all PyTorch cuda caches
            for i in range(torch.cuda.device_count()):
                torch.cuda.memory.empty_cache()
                torch.cuda.memory.reset_peak_memory_stats()
            logger.info("Reset all PyTorch CUDA memory stats")
        
        # Clear Python object cache
        gc.collect(generation=2)
        gc.collect(generation=2)
        
        # On Render, look for large objects to clear
        if ON_RENDER:
            self._identify_and_remove_large_objects()
        
        logger.info("Completed aggressive cleanup")
    
    def _identify_and_remove_large_objects(self):
        """Identify and attempt to remove references to large objects"""
        try:
            # This is a simple approach to finding large objects
            # For production use, consider using objgraph or memory_profiler
            logger.info("Looking for large objects to clean up")
            
            # Collect all objects
            gc.collect(generation=2)
            
            # This is a last resort - consider more targeted cleanup in your actual app
            import sys as _sys
            if hasattr(_sys, "getsizeof"):
                threshold = 1024 * 1024 * 10  # 10MB
                count = 0
                for obj in gc.get_objects():
                    try:
                        if _sys.getsizeof(obj) > threshold:
                            count += 1
                            del obj
                    except:
                        pass
                if count > 0:
                    logger.info(f"Identified and removed references to {count} large objects")
                    gc.collect(generation=2)
        except Exception as e:
            logger.error(f"Error during large object cleanup: {e}")
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup - take drastic measures"""
        logger.critical("Performing EMERGENCY memory cleanup")
        
        # Aggressive cleanup first
        self._aggressive_cleanup()
        
        # More drastic measures
        # Clear all module-level caches that might be using memory
        self._clear_module_caches()
        
        # Run garbage collection several times
        for _ in range(3):
            gc.collect(generation=2)
        
        # On Render with critical memory, take more drastic measures
        if ON_RENDER:
            try:
                # Try to release as much memory as possible on Render
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
                
                # Clear known caches in libraries
                if 'sentence_transformers' in sys.modules:
                    # Clear sentence transformer caches if possible
                    import importlib
                    importlib.reload(sys.modules['sentence_transformers'])
                
                # Last resort: manually trigger Python's own memory deallocation
                for _ in range(3):
                    gc.collect(generation=2)
            except Exception as e:
                logger.error(f"Error during emergency memory deallocation: {e}")
        
        logger.critical("Completed emergency cleanup")
    
    def _clear_module_caches(self):
        """Clear caches in various modules that might be using memory"""
        # Clear lru_caches and other common caches
        try:
            # Try to clear functools cache if used
            import functools
            if hasattr(functools, "_CacheInfo"):
                for module_name in list(sys.modules.keys()):
                    module = sys.modules[module_name]
                    for attr_name in dir(module):
                        try:
                            attr = getattr(module, attr_name)
                            if hasattr(attr, "cache_clear") and callable(attr.cache_clear):
                                attr.cache_clear()
                        except:
                            pass
            
            logger.info("Cleared function caches in modules")
        except Exception as e:
            logger.error(f"Error clearing module caches: {e}")
    
    def start_monitoring(self):
        """Start monitoring in a separate thread"""
        if self.running:
            logger.warning("Memory monitor is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started memory monitoring thread (every {self.interval} seconds)")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Stopped memory monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            self.log_memory_usage()
            time.sleep(self.interval)

def main():
    """Main function to run the memory monitor as a standalone script"""
    parser = argparse.ArgumentParser(description="Memory monitor for Resume Scorer")
    parser.add_argument("--interval", type=int, default=MEMORY_MONITORING_INTERVAL,
                        help=f"Monitoring interval in seconds (default: {MEMORY_MONITORING_INTERVAL})")
    parser.add_argument("--one_time", action="store_true",
                        help="Run once and exit (don't start monitoring loop)")
    args = parser.parse_args()
    
    monitor = MemoryMonitor(interval=args.interval)
    
    if args.one_time:
        # Just log once and exit
        monitor.log_memory_usage()
    else:
        # Start continuous monitoring
        try:
            logger.info("Starting memory monitor")
            monitor.start_monitoring()
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping memory monitor due to keyboard interrupt")
            monitor.stop_monitoring()
        except Exception as e:
            logger.error(f"Error in memory monitor: {e}")
            monitor.stop_monitoring()

if __name__ == "__main__":
    main() 