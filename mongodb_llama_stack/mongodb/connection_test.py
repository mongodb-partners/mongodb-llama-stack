#!/usr/bin/env python3
"""
MongoDB Connection Test

This script tests the connection to MongoDB and identifies available features.
"""

import os
import sys
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import certifi

def test_connection():
    """Test the MongoDB connection and report on available features."""
    # Get connection string from environment
    connection_string = os.environ.get('MONGODB_CONNECTION_STR')
    if not connection_string:
        print("Error: MONGODB_CONNECTION_STR environment variable not set.")
        print("Please set it to your MongoDB connection string.")
        print("Example: export MONGODB_CONNECTION_STR='mongodb+srv://username:password@cluster.mongodb.net/'")
        return False
    
    try:
        # Create a MongoDB client
        client = MongoClient(connection_string, tlsCAFile=certifi.where())
        
        # Test the connection
        client.admin.command('ping')
        
        # Get server info
        server_info = client.admin.command('buildInfo')
        server_version = server_info.get('version', 'unknown')
        
        # Check for feature compatibility
        features = []
        
        # Check for Atlas
        try:
            is_atlas = 'atlas' in connection_string or client.admin.command({'getParameter': 1, 'featureCompatibilityVersion': 1})
            if is_atlas:
                features.append("Atlas")
        except Exception:
            pass
        
        # Check for vector search capability
        try:
            # Try to validate a simple vector search pipeline
            client.admin.command({
                'aggregate': 1,
                'pipeline': [
                    {'$vectorSearch': {'index': 'test', 'path': 'embedding', 'queryVector': [0.1], 'numCandidates': 10, 'limit': 5}}
                ],
                'cursor': {}
            })
            features.append("vectorSearch")
        except OperationFailure:
            pass
        
        # Check for text search capability
        try:
            client.admin.command({
                'aggregate': 1, 
                'pipeline': [{'$search': {'text': {'query': 'test', 'path': 'content'}}}],
                'cursor': {}
            })
            features.append("search")
        except OperationFailure:
            pass
        
        # Check for rank fusion capability (MongoDB 8.1+)
        try:
            client.admin.command({
                'aggregate': 1,
                'pipeline': [{'$rankFusion': {'clauses': []}}],
                'cursor': {}
            })
            features.append("rankFusion")
        except OperationFailure:
            pass
        
        # Check for graph lookup capability
        try:
            client.admin.command({
                'aggregate': 1,
                'pipeline': [{'$graphLookup': {'from': 'test', 'startWith': '$field', 'connectFromField': 'field', 'connectToField': 'field', 'as': 'result'}}],
                'cursor': {}
            })
            features.append("graphLookup")
        except OperationFailure:
            pass
            
        # Print connection results
        print("\nMongoDB connection successful! 🎉")
        print(f"Server version: {server_version}")
        print(f"Available features: {', '.join(features)}")
        
        # Check namespace if provided
        namespace = os.environ.get('MONGODB_NAMESPACE')
        if namespace:
            try:
                db_name, collection_name = namespace.split('.')
                collection = client[db_name][collection_name]
                doc_count = collection.count_documents({})
                print(f"Collection '{namespace}' contains {doc_count} documents")
                
                # Check for existing search indexes
                try:
                    indexes = list(collection.list_search_indexes())
                    if indexes:
                        print(f"Found {len(indexes)} search indexes:")
                        for idx in indexes:
                            print(f"  - {idx['name']} (type: {idx.get('type', 'unknown')})")
                    else:
                        print("No search indexes found on collection")
                except Exception as e:
                    print(f"Could not list search indexes: {str(e)}")
                    
            except ValueError:
                print(f"Warning: Invalid namespace format: {namespace}. Expected format: 'database.collection'")
            except Exception as e:
                print(f"Warning: Could not access namespace {namespace}: {str(e)}")
        
        return True
        
    except ConnectionFailure:
        print("Error: Failed to connect to MongoDB. Please check your connection string.")
        return False
    except OperationFailure as e:
        print(f"Error: MongoDB operation failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
