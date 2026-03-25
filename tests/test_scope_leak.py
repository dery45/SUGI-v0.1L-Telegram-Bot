import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))

from core.sugi_core import SugiCore

def test():
    sugi = SugiCore()
    uid  = "test_scope_fix"

    print("="*60)
    print("TEST 1: Normal agriculture query")
    print("="*60)
    q1 = "tanah yang cocok untuk tanaman semangka"
    print(f"User: {q1}")
    r1 = sugi.ask(uid, q1)
    print(f"Sugi: {r1[:200]}...\n")

    print("="*60)
    print("TEST 2: Referential followup (suffix-based rewrite)")
    print("="*60)
    q2 = "apakah sekarang musim yang cocok untuk menanamnya"
    print(f"User: {q2}")
    r2 = sugi.ask(uid, q2)
    print(f"Sugi: {r2[:200]}...\n")

    print("="*60)
    print("TEST 3: Out-of-scope with 'itu' (definitional)")
    print("="*60)
    q3 = "apa itu machine learning"
    print(f"User: {q3}")
    r3 = sugi.ask(uid, q3)
    is_refused = (r3 == sugi.refusal_msg)
    print(f"Sugi: {r3[:200]}")
    if is_refused:
        print("\n✅ TEST 3 PASSED: Correctly blocked out-of-scope query!")
    else:
        print("\n❌ TEST 3 FAILED: Should have been blocked!")

    print("\n" + "="*60)
    print("TEST 4: 'what is photosynthesis' (definitional, in-scope)")
    print("="*60)
    q4 = "apa itu fotosintesis"
    print(f"User: {q4}")
    r4 = sugi.ask(uid, q4)
    is_refused4 = (r4 == sugi.refusal_msg)
    print(f"Sugi: {r4[:200]}")
    if not is_refused4:
        print("\n✅ TEST 4 PASSED: Allowed in-scope definitional query!")
    else:
        print("\n⚠️  TEST 4: Blocked — check if 'fotosintesis' is in scope keywords")

if __name__ == "__main__":
    test()
