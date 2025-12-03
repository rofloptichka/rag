"""
Script to upload a document and check the markdown result from Docling conversion.
This helps you verify the markdown output quality before full indexing.
"""

import requests
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_BASE_URL = "http://localhost:8000"  # Change to your API URL if different
COMPANY_ID = "test-company"  # Replace with your actual company ID
FILE_PATH = None  # Will be set from command line or config

# ==============================================================================
# MARKDOWN PREVIEW FUNCTION
# ==============================================================================
def print_markdown_preview(markdown_text: str, max_length: int = 2000):
    """
    Display markdown with nice formatting in terminal
    """
    print("\n" + "=" * 80)
    print("📄 MARKDOWN PREVIEW")
    print("=" * 80)
    
    if len(markdown_text) > max_length:
        print(f"\n[Showing first {max_length} characters of {len(markdown_text)} total]\n")
        print(markdown_text[:max_length])
        print(f"\n... ({len(markdown_text) - max_length} more characters)")
    else:
        print(f"\n[Total length: {len(markdown_text)} characters]\n")
        print(markdown_text)
    
    print("\n" + "=" * 80)


def save_markdown_to_file(markdown_text: str, filename: str, output_dir: str = "markdown_outputs"):
    """
    Save markdown result to a file for further inspection
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename based on input
    base_name = Path(filename).stem
    output_path = os.path.join(output_dir, f"{base_name}_output.md")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    
    print(f"✅ Markdown saved to: {output_path}")
    return output_path


def upload_and_check_markdown(
    file_path: str,
    company_id: str,
    stream: bool = False,
    save_to_file: bool = True,
    show_preview_length: int = 2000
) -> Optional[dict]:
    """
    Upload a document and check its markdown conversion without storing in Qdrant.
    
    Args:
        file_path: Full path to the file
        company_id: Your company ID
        stream: Use streaming for real-time updates
        save_to_file: Save markdown output to file
        show_preview_length: Max length to show in terminal
    
    Returns:
        Dictionary with markdown and metadata, or None if failed
    """
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    print("\n" + "=" * 80)
    print("🚀 DOCUMENT MARKDOWN CHECKER")
    print("=" * 80)
    print(f"📁 File: {filename}")
    print(f"📊 Size: {file_size:,} bytes")
    print(f"🔗 Endpoint: {API_BASE_URL}/companies/{company_id}/process-document")
    print(f"🌊 Streaming: {stream}")
    print("-" * 80)
    
    endpoint = f"{API_BASE_URL}/companies/{company_id}/process-document"
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        
        params = {
            "stream": stream,
            "llmEnrich": False,  # Don't need LLM for just checking markdown
            "naturalChunking": False,  # Don't need chunking for just checking markdown
            "allowUpdate": True,  # Allow re-upload for testing
        }
        
        try:
            print("📤 Uploading and processing document...\n")
            response = requests.post(endpoint, files=files, params=params, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ Upload successful!\n")
                print(f"Response metadata:")
                print(f"  - Chunks created: {result.get('row_count', 0)}")
                print(f"  - GCS URL: {result.get('url', 'N/A')}")
                print(f"  - LLM enriched: {result.get('llm', False)}")
                print(f"  - Natural chunking: {result.get('natural', False)}")
                
                return result
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout (>300s). File might be too large or API is slow.")
            return None
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            return None


def upload_and_check_markdown_with_streaming(
    file_path: str,
    company_id: str
) -> Optional[dict]:
    """
    Upload with streaming to see real-time markdown preview
    """
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    filename = os.path.basename(file_path)
    
    print("\n" + "=" * 80)
    print("🚀 DOCUMENT MARKDOWN CHECKER (STREAMING)")
    print("=" * 80)
    print(f"📁 File: {filename}")
    print(f"🔗 Endpoint: {API_BASE_URL}/companies/{company_id}/process-document")
    print("-" * 80)
    
    endpoint = f"{API_BASE_URL}/companies/{company_id}/process-document"
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        params = {
            "stream": True,  # Enable streaming
            "llmEnrich": False,
            "naturalChunking": False,
            "allowUpdate": True,
        }
        
        print("📤 Uploading with real-time updates...\n")
        
        try:
            response = requests.post(endpoint, files=files, params=params, stream=True, timeout=300)
            
            if response.status_code == 200:
                markdown_preview = None
                final_result = None
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                                
                                event_type = data.get("type", "")
                                payload = data.get("payload", {})
                                
                                if event_type == "status":
                                    step = payload.get("step", "")
                                    message = payload.get("message", "")
                                    print(f"📨 [{step.upper()}] {message}")
                                
                                elif event_type == "debug":
                                    if "markdown_preview" in payload:
                                        markdown_preview = payload["markdown_preview"]
                                        print(f"\n📑 Markdown preview (first 500 chars):\n")
                                        print(markdown_preview[:500])
                                        if len(markdown_preview) > 500:
                                            print(f"\n... ({len(markdown_preview) - 500} more characters)")
                                    
                                    if "sections_count" in payload:
                                        print(f"📊 Sections extracted: {payload['sections_count']}")
                                    if "natural_chunks" in payload:
                                        print(f"📊 Natural chunks: {payload['natural_chunks']}")
                                    if "hybrid_chunks" in payload:
                                        print(f"📊 Hybrid chunks: {payload['hybrid_chunks']}")
                                
                                elif event_type == "done":
                                    final_result = payload
                                    print(f"\n✅ Processing complete!")
                                    print(f"   - Chunks: {payload.get('row_count', 0)}")
                                
                                elif event_type == "error":
                                    print(f"❌ Error: {payload.get('message', 'Unknown error')}")
                            
                            except json.JSONDecodeError:
                                print(f"📨 {line_str}")
                
                print("\n" + "=" * 80)
                return final_result
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout (>300s). File might be too large.")
            return None
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            return None


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================
def main():
    """
    Main entry point with command-line argument handling
    """
    
    print("\n" + "=" * 80)
    print("📝 DOCUMENT MARKDOWN CHECKER")
    print("=" * 80)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\n❌ Usage: python check_markdown.py <file_path> [company_id] [--stream] [--save]")
        print("\nExamples:")
        print("  python check_markdown.py C:\\docs\\document.pdf")
        print("  python check_markdown.py C:\\docs\\document.pdf my-company --stream --save")
        print("  python check_markdown.py C:\\docs\\document.docx test-company --stream")
        print("\nOptions:")
        print("  --stream    : Use Server-Sent Events streaming for real-time updates")
        print("  --save      : Save markdown output to file in 'markdown_outputs' folder")
        return
    
    file_path = sys.argv[1]
    company_id = sys.argv[2] if len(sys.argv) > 2 else COMPANY_ID
    use_stream = "--stream" in sys.argv
    save_to_file = "--save" in sys.argv
    
    print(f"\nConfiguration:")
    print(f"  API Base URL: {API_BASE_URL}")
    print(f"  Company ID: {company_id}")
    print(f"  File: {file_path}")
    print(f"  Streaming: {use_stream}")
    print(f"  Save to file: {save_to_file}")
    
    # Run the check
    if use_stream:
        result = upload_and_check_markdown_with_streaming(file_path, company_id)
    else:
        result = upload_and_check_markdown(file_path, company_id, save_to_file=save_to_file)
    
    if result:
        print(f"\n✅ SUCCESS - Document processed successfully")
        print(f"Total chunks: {result.get('row_count', 0)}")
    else:
        print(f"\n❌ FAILED - Could not process document")
        sys.exit(1)


if __name__ == "__main__":
    main()
