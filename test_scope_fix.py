from core.sugi_core import SugiCore
import os

def test_scope():
    sugi = SugiCore()
    user_id = "test_user"
    
    # Pre-load context
    print("\n--- Context: semangka ---")
    sugi.ask("tanah yang cocok untuk tanaman semangka", user_id)
    
    test_cases = [
        "bagaimana merawat tanamanya ?",
        "apakah sekarang musim yang cocok untuk menanamnya",
        "siapa kamu ?",
        "apa itu sugi?"
    ]
    
    for q in test_cases:
        print(f"\nTesting: '{q}'")
        response = sugi.ask(q, user_id)
        if response == sugi.refusal_msg:
            print(f"❌ REJECTED: {q}")
        else:
            print(f"✅ ALLOWED: {q}")
            # print(f"Response snippet: {response[:100]}...")

if __name__ == "__main__":
    test_scope()
