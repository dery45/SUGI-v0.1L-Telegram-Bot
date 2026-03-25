import sys
import os
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

from core.sugi_core import SugiCore

def test_query_flow():
    sugi = SugiCore()
    user_id = "test_user_fix"
    
    print("\n--- Testing Query 1 ---")
    q1 = "tanah yang cocok untuk tanaman semangka"
    print(f"User: {q1}")
    a1 = sugi.ask(user_id, q1)
    # print(f"Sugi: {a1[:100]}...")
    
    print("\n--- Testing Query 2 (Referential) ---")
    q2 = "apakah sekarang musim yang cocok untuk menanamnya"
    print(f"User: {q2}")
    a2 = sugi.ask(user_id, q2)
    # print(f"Sugi: {a2[:100]}...")
    
    # Check if we got an error message or a real answer
    if "Maaf, terjadi kesalahan" in a2:
        print("\n❌ FAILED: Still getting error in Query 2")
    else:
        print("\n✅ SUCCESS: No crash in Query 2")

if __name__ == "__main__":
    test_query_flow()
