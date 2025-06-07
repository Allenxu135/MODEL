import os
import psutil
import time
import requests
import subprocess
from pathlib import Path
import ollama

# Constants
OLLAMA_SIGNATURE = {'manifests', 'blobs'}
GGUF_EXTENSION = '.gguf'
SKIP_DIRS = {
    'Windows', 'Program Files', 'System32', 'System Volume Information',
    '$RECYCLE.BIN', 'AppData', 'tmp', 'Temp', 'Library', 'etc', 'dev'
}
API_URL = "http://localhost:11434/api/generate"
SCAN_TIMEOUT = 5
API_TIMEOUT = 10


def get_ollama_paths():
    """Get all valid Ollama model paths with priority order"""
    paths = set()

    # Environment variable path
    if env_path := os.environ.get("OLLAMA_MODELS"):
        if (custom_path := Path(env_path)).exists():
            paths.add(custom_path)

    # Platform defaults
    platform_paths = [
        Path.home() / ".ollama" / "models",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
    ]

    # Custom locations
    custom_locations = [
        Path("D:\AIMODELS"),
        Path("E:/AI_Models"),
        Path("/mnt/ai_models")
    ]

    # Add valid paths with existence check
    for path in [*platform_paths, *custom_locations]:
        if path.exists():
            paths.add(path.resolve())  # Normalize path

    return list(paths)


def is_skip_dir(root_path):
    """Check if directory should be skipped during scanning"""
    return any(
        skip_dir in root_path or
        root_path.startswith('.')  # Skip hidden directories
        for skip_dir in SKIP_DIRS
    )


def scan_all_drives_for_models():
    """Efficiently scan all drives for Ollama-compatible models"""
    model_dirs = set(get_ollama_paths())

    for partition in psutil.disk_partitions(all=True):
        try:
            for root, dirs, files in os.walk(partition.mountpoint, topdown=True):
                # Prune search tree for efficiency
                if is_skip_dir(root):
                    dirs[:] = []
                    continue

                # Check for Ollama structure
                if OLLAMA_SIGNATURE.issubset(set(os.listdir(root))):
                    model_dirs.add(root)
                    dirs[:] = []  # Don't descend further
                    continue

                # Check for GGUF files
                if any(f.endswith(GGUF_EXTENSION) for f in files):
                    model_dirs.add(root)

        except Exception as e:
            print(f"⚠️ Scan error: {str(e)}")

    return list(model_dirs)


def test_ollama_connection(model_name=""):
    """Robust API connection test with proper error handling"""
    try:
        # First try Python package method
        try:
            if model_list := ollama.list().get('models'):
                model_match = next(
                    (m for m in model_list if not model_name or m['name'].startswith(model_name)),
                    None
                )
                if model_match:
                    return True, f"API connected (Model: {model_match['name']})"
        except Exception as e:
            print(f"⚠️ Package error: {str(e)}")

        # Fallback to direct API request
        response = requests.post(
            API_URL,
            json={"model": model_name or "llama3", "prompt": "Connection test", "stream": False},
            timeout=API_TIMEOUT
        )

        if response.status_code == 200:
            return True, "API connected via direct request"
        return False, f"API error: {response.status_code} {response.text}"

    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def identify_models(path):
    """Accurately identify models in a directory"""
    path = Path(path)
    models = []

    # Detect Ollama internal models
    manifests = path / "manifests"
    if manifests.exists():
        models.extend(f"Ollama Model: {d.name}" for d in manifests.iterdir() if d.is_dir())

    # Detect GGUF files
    models.extend(
        f"GGUF Model: {f.stem}"
        for f in path.glob(f"*{GGUF_EXTENSION}")
        if f.is_file()
    )

    return models or ["Unknown model format"]


def start_ollama_service():
    """Ensure Ollama service is running"""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        time.sleep(2)  # Allow service initialization
    except Exception as e:
        print(f"⚠️ Failed to start Ollama: {str(e)}")


def main():
    """Main execution flow with improved structure"""
    print("🔍 Scanning system for Ollama models...")
    start_time = time.time()

    # Ensure service is running
    start_ollama_service()

    # Scan for models
    model_dirs = scan_all_drives_for_models()

    if not model_dirs:
        print("\n" + "=" * 50)
        print("❌ NO MODELS DETECTED")
        print("=" * 50)
        return

    print(f"\nFound {len(model_dirs)} model locations")

    # Test API connection
    print("\n" + "=" * 50)
    print("🧪 Testing Ollama API Connection")
    print("=" * 50)
    api_status, api_msg = test_ollama_connection()
    print(f"Status: {'✅' if api_status else '❌'} {api_msg}")
    print("=" * 50)

    # Process locations
    for idx, model_dir in enumerate(model_dirs, 1):
        print(f"\n📍 Location #{idx}: {model_dir}")
        models = identify_models(model_dir)

        print(f"  - Models detected: {len(models)}")
        for model in models:
            print(f"    • {model}")

            # Test identifiable models
            if model.startswith(("Ollama Model:", "GGUF Model:")):
                model_name = model.split(":", 1)[1].strip()
                print(f"\n  🧪 Testing model: {model_name}")
                status, msg = test_ollama_connection(model_name)
                print(f"  - Test result: {'✅' if status else '❌'} {msg}")

    print(f"\nTotal scan time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()