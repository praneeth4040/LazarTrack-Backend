#!/usr/bin/env python3
"""
Quick test script to verify Supabase Storage connection and upload permissions
"""

import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.supabase_storage import storage_client

def test_storage_connection():
    """Test that Supabase Storage client initializes correctly"""
    print("=" * 60)
    print("Testing Supabase Storage Connection")
    print("=" * 60)
    
    # Check credentials
    from app.services.supabase_storage import SUPABASE_URL, SUPABASE_ANON_KEY, BUCKET_NAME
    
    print(f"\n✓ URL: {SUPABASE_URL}")
    print(f"✓ Key (first 20 chars): {SUPABASE_ANON_KEY[:20]}...")
    print(f"✓ Bucket: {BUCKET_NAME}")
    
    # Check HTTP client initialization
    if not storage_client.base_url or not storage_client.anon_key:
        print("\n✗ Failed to initialize Supabase client (missing credentials)!")
        return False
    
    print("\n✓ Supabase HTTP client initialized successfully")
    
    # Test upload with a simple text file
    print("\n" + "-" * 60)
    print("Testing File Upload")
    print("-" * 60)
    
    test_content = b"Test content from Python\n"
    test_path = "test_upload/test_file.txt"
    
    try:
        print(f"\nUploading test file to: {test_path}")
        result = storage_client.upload_file_from_bytes(
            test_content,
            test_path,
            content_type="text/plain"
        )
        
        if result:
            print(f"✓ Upload successful!")
            print(f"  URL: {result}")
            
            # Try to get public URL
            public_url = storage_client.get_public_url(test_path)
            print(f"  Public URL: {public_url}")
            
            # Cleanup
            print(f"\nCleaning up test file...")
            storage_client.delete_file(test_path)
            print("✓ Test file deleted")
            
            return True
        else:
            print("✗ Upload failed (no URL returned)")
            return False
            
    except Exception as e:
        print(f"✗ Upload test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_storage_connection()
    sys.exit(0 if success else 1)
