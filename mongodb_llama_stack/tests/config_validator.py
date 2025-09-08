#!/usr/bin/env python3
"""
Simple configuration validator for MongoDB Llama Stack provider
"""

import sys
import os
import logging
from typing import Optional, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)

def check_environment_variables() -> Dict[str, Any]:
    """Check required and optional environment variables"""
    required_vars = ["MONGODB_CONNECTION_STR"]
    optional_vars = [
        "MONGODB_NAMESPACE", 
        "MONGODB_SEARCH_MODE",
        "MONGODB_INDEX_NAME",
        "MONGODB_TEXT_INDEX_NAME",
        "EXTERNAL_PROVIDERS_DIR"
    ]
    
    config = {}
    missing_vars = []
    
    # Check required variables
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
        else:
            config[var] = value
    
    # Check optional variables
    for var in optional_vars:
        value = os.environ.get(var)
        if value:
            config[var] = value
    
    # Report missing required variables
    if missing_vars:
        log.error("Missing required environment variables: %s", ", ".join(missing_vars))
        return config
    
    log.info("Environment variables validated successfully")
    return config

def validate_connection_string(connection_str: str) -> bool:
    """Validate MongoDB connection string format"""
    if not connection_str:
        return False
    
    valid_prefixes = [
        "mongodb://",
        "mongodb+srv://"
    ]
    
    if not any(connection_str.startswith(prefix) for prefix in valid_prefixes):
        log.error("Invalid connection string format. Should start with mongodb:// or mongodb+srv://")
        return False
    
    log.info("Connection string format is valid")
    return True

def validate_namespace(namespace: Optional[str]) -> bool:
    """Validate MongoDB namespace format (db.collection)"""
    if not namespace:
        log.warning("No namespace provided, will use default")
        return True
    
    parts = namespace.split(".")
    if len(parts) != 2 or not all(parts):
        log.error("Invalid namespace format. Should be 'database.collection'")
        return False
    
    log.info("Namespace format is valid: %s", namespace)
    return True

def validate_search_mode(search_mode: Optional[str]) -> bool:
    """Validate search mode value"""
    if not search_mode:
        log.info("No search mode specified, will use default (vector)")
        return True
    
    valid_modes = [
        "vector", 
        "full_text", 
        "hybrid", 
        "native_rank_fusion",
        "hybrid_graph"
    ]
    
    if search_mode.lower() not in valid_modes:
        log.error("Invalid search mode. Must be one of: %s", ", ".join(valid_modes))
        return False
    
    log.info("Search mode is valid: %s", search_mode)
    return True

def validate_providers_dir(providers_dir: Optional[str]) -> bool:
    """Validate external providers directory"""
    if not providers_dir:
        log.warning("No EXTERNAL_PROVIDERS_DIR specified")
        return True
    
    if not os.path.isdir(providers_dir):
        log.error("EXTERNAL_PROVIDERS_DIR does not exist: %s", providers_dir)
        return False
    
    # Check for mongodb.yaml provider file
    provider_file = os.path.join(providers_dir, "remote", "vector_io", "mongodb.yaml")
    if not os.path.isfile(provider_file):
        log.warning("Provider discovery file not found: %s", provider_file)
        return False
    
    log.info("External providers directory is valid: %s", providers_dir)
    log.info("Provider discovery file found: %s", provider_file)
    return True

def main():
    """Main validation function"""
    log.info("MongoDB Llama Stack provider configuration validator")
    log.info("=" * 50)
    
    # Check environment variables
    config = check_environment_variables()
    
    # Validate connection string
    connection_valid = validate_connection_string(config.get("MONGODB_CONNECTION_STR", ""))
    
    # Validate namespace
    namespace_valid = validate_namespace(config.get("MONGODB_NAMESPACE"))
    
    # Validate search mode
    search_mode_valid = validate_search_mode(config.get("MONGODB_SEARCH_MODE"))
    
    # Validate providers directory
    providers_dir_valid = validate_providers_dir(config.get("EXTERNAL_PROVIDERS_DIR"))
    
    # Print configuration summary
    log.info("\nConfiguration Summary:")
    log.info("-" * 50)
    
    for key, value in config.items():
        if key == "MONGODB_CONNECTION_STR":
            # Mask connection string for security
            masked = value[:15] + "..." + value[-10:] if len(value) > 30 else "..."
            log.info("%s: %s", key, masked)
        else:
            log.info("%s: %s", key, value)
    
    # Print validation summary
    log.info("\nValidation Results:")
    log.info("-" * 50)
    
    all_valid = all([
        connection_valid, 
        namespace_valid, 
        search_mode_valid,
        providers_dir_valid
    ])
    
    if all_valid:
        log.info("✅ All configuration checks passed")
    else:
        log.warning("⚠️ Some configuration checks failed, see details above")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())
