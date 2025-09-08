#!/usr/bin/env python3
"""
Connection test for MongoDB Llama Stack provider
"""

import asyncio
import sys
import os
import logging
import time
import pymongo
from pymongo import MongoClient
from packaging import version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)

class MongoDBConnectionTest:
    """Test MongoDB connection and capabilities"""
    
    def __init__(self):
        """Initialize with connection details from environment"""
        self.connection_str = os.environ.get("MONGODB_CONNECTION_STR")
        self.namespace = os.environ.get("MONGODB_NAMESPACE", "test_llama.connection_test")
        
        if not self.connection_str:
            raise ValueError("MONGODB_CONNECTION_STR environment variable is required")
        
        # Split namespace into database and collection
        parts = self.namespace.split(".")
        if len(parts) != 2:
            raise ValueError("MONGODB_NAMESPACE should be in format 'database.collection'")
        
        self.db_name = parts[0]
        self.collection_name = parts[1]
        
        # Connection attributes
        self.client = None
        self.is_atlas = False
        self.server_version = None
        self.supports_rank_fusion = False
        self.supports_vector_search = False
    
    async def test_connection(self):
        """Test basic connectivity to MongoDB"""
        try:
            log.info("Testing MongoDB connection...")
            
            start_time = time.time()
            self.client = MongoClient(self.connection_str)
            
            # Test connection with ping
            self.client.admin.command('ping')
            
            end_time = time.time()
            log.info("✅ Connection successful (%.2f ms)", (end_time - start_time) * 1000)
            return True
            
        except Exception as e:
            log.error("❌ Connection failed: %s", e)
            return False
    
    async def check_server_version(self):
        """Check MongoDB server version and capabilities"""
        if not self.client:
            log.error("No connection available")
            return False
        
        try:
            server_info = self.client.server_info()
            self.server_version = server_info.get('version', '0.0.0')
            log.info("MongoDB server version: %s", self.server_version)
            
            # Check version features
            v = version.parse(self.server_version)
            self.supports_vector_search = v >= version.parse('7.0.0')
            self.supports_rank_fusion = v >= version.parse('8.1.0')
            
            # Check if running on Atlas
            self.is_atlas = ('mongodb.net' in self.connection_str or 
                           'mongodb-dev.net' in self.connection_str)
            
            # Log capabilities
            log.info("Atlas deployment: %s", "Yes" if self.is_atlas else "No")
            log.info("Vector search support: %s", "Yes" if self.supports_vector_search else "No")
            log.info("Native rank fusion support: %s", "Yes" if self.supports_rank_fusion else "No")
            
            return True
            
        except Exception as e:
            log.error("❌ Failed to get server information: %s", e)
            return False
    
    async def test_database_access(self):
        """Test database read/write access"""
        if not self.client:
            log.error("No connection available")
            return False
        
        try:
            # Access database and collection
            db = self.client[self.db_name]
            collection = db[self.collection_name]
            
            # Insert test document
            log.info("Testing write access...")
            test_doc = {
                "test_id": "connection_test",
                "timestamp": int(time.time()),
                "message": "MongoDB Llama Stack provider connection test"
            }
            
            result = collection.insert_one(test_doc)
            log.info("✅ Document inserted with ID: %s", result.inserted_id)
            
            # Read the document back
            log.info("Testing read access...")
            found_doc = collection.find_one({"test_id": "connection_test"})
            
            if found_doc:
                log.info("✅ Document retrieved successfully")
                
                # Clean up - delete the test document
                collection.delete_one({"_id": found_doc["_id"]})
                log.info("✅ Test document cleaned up")
                
                return True
            else:
                log.error("❌ Could not retrieve test document")
                return False
                
        except pymongo.errors.OperationFailure as e:
            log.error("❌ Database operation failed: %s", e)
            
            if "not authorized" in str(e):
                log.error("   Permission error - check your connection string and user permissions")
            
            return False
            
        except Exception as e:
            log.error("❌ Database access test failed: %s", e)
            return False
    
    async def test_search_indexes(self):
        """Test creating and using search indexes if on Atlas"""
        if not self.client or not self.is_atlas:
            if not self.is_atlas:
                log.info("Skipping search index test (not on Atlas)")
            return True
        
        try:
            db = self.client[self.db_name]
            collection = db[self.collection_name]
            
            # Try to create a basic search index
            log.info("Testing Atlas Search index creation...")
            
            # First, check if indexes already exist
            try:
                indexes = list(collection.list_search_indexes())
                log.info("Found %d existing search indexes", len(indexes))
                for idx in indexes:
                    log.info("  - %s", idx.get("name", "unnamed"))
            except Exception as e:
                log.warning("Could not list search indexes: %s", e)
            
            # Test creating a simple index
            try:
                from pymongo.operations import SearchIndexModel
                
                # Create simple text index model
                text_index = SearchIndexModel(
                    name="test_search_index",
                    type="search",
                    definition={
                        "mappings": {
                            "dynamic": True
                        }
                    }
                )
                
                # Try to create (this may fail if user doesn't have permissions)
                collection.create_search_index(text_index)
                log.info("✅ Search index created successfully")
                
                # Clean up - drop the test index
                collection.drop_search_index("test_search_index")
                log.info("✅ Test search index cleaned up")
                
                return True
                
            except pymongo.errors.OperationFailure as e:
                if "not authorized" in str(e):
                    log.warning("⚠️ Search index creation requires additional permissions")
                    log.warning("   This is expected if using a restricted user")
                    return True
                else:
                    log.warning("⚠️ Search index creation failed: %s", e)
                    return True
                    
        except Exception as e:
            log.warning("⚠️ Search index test encountered an error: %s", e)
            return True  # Non-fatal
    
    async def run_all_tests(self):
        """Run all connection tests"""
        log.info("MongoDB Connection Test Suite")
        log.info("=" * 50)
        log.info("Connection string: %s", self.connection_str[:15] + "..." + self.connection_str[-10:] 
                 if len(self.connection_str) > 30 else "...")
        log.info("Namespace: %s", self.namespace)
        
        # Run tests sequentially
        connection_ok = await self.test_connection()
        if not connection_ok:
            log.error("Connection test failed - aborting further tests")
            return False
        
        version_ok = await self.check_server_version()
        access_ok = await self.test_database_access()
        search_ok = await self.test_search_indexes()
        
        # Print summary
        log.info("\nTest Summary:")
        log.info("-" * 50)
        log.info("Connection:       %s", "✅ Passed" if connection_ok else "❌ Failed")
        log.info("Version Check:    %s", "✅ Passed" if version_ok else "❌ Failed")
        log.info("Database Access:  %s", "✅ Passed" if access_ok else "❌ Failed")
        log.info("Search Indexes:   %s", "✅ Passed" if search_ok else "⚠️ Limited")
        
        # Provider compatibility check
        provider_compatible = connection_ok and version_ok and access_ok
        
        if provider_compatible:
            log.info("\n✅ MongoDB connection is READY for use with Llama Stack")
            
            # Print feature compatibility
            log.info("\nFeature Compatibility:")
            log.info("-" * 50)
            log.info("Vector Search:       %s", "✅ Available" if self.supports_vector_search else "❌ Unavailable (requires MongoDB 7.0+)")
            log.info("Native Rank Fusion:  %s", "✅ Available" if self.supports_rank_fusion else "❌ Unavailable (requires MongoDB 8.1+)")
            log.info("Atlas Search:        %s", "✅ Available" if self.is_atlas else "❌ Unavailable (requires MongoDB Atlas)")
            
            search_modes = ["vector"]
            if self.is_atlas:
                search_modes.append("full_text")
                search_modes.append("hybrid")
            if self.supports_rank_fusion:
                search_modes.append("native_rank_fusion")
            
            log.info("\nAvailable search modes: %s", ", ".join(search_modes))
            
        else:
            log.error("\n❌ MongoDB connection is NOT READY for use with Llama Stack")
            log.error("   Please fix the issues above before proceeding")
        
        return provider_compatible

async def main():
    """Main entry point"""
    try:
        tester = MongoDBConnectionTest()
        success = await tester.run_all_tests()
        return 0 if success else 1
        
    except ValueError as e:
        log.error("❌ Configuration error: %s", e)
        log.error("\nPlease set the required environment variables:")
        log.error("export MONGODB_CONNECTION_STR='mongodb+srv://username:password@cluster.mongodb.net/'")
        log.error("export MONGODB_NAMESPACE='database.collection'")
        return 1
        
    except KeyboardInterrupt:
        log.info("\nTest interrupted")
        return 1
        
    except Exception as e:
        log.error("❌ Unexpected error: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
