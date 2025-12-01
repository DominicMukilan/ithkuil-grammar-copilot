"""
Experiment: Measure LLM Accuracy With vs Without RAG
====================================================

This generates the actual numbers for your portfolio.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from copilot import CopilotPipeline


# Test cases with KNOWN correct answers
# Format: (english_input, correct_case, correct_function, explanation)
TEST_CASES: List[Tuple[str, str, str, str]] = [
    # AFF cases (EXPERIENCER) - must use STA
    ("experiencing cold involuntarily", "AFF", "STA", "unwilled bodily sensation"),
    ("feeling fear", "AFF", "STA", "unwilled emotion"),
    ("sneezing", "AFF", "STA", "involuntary reflex"),
    ("hearing a loud noise", "AFF", "STA", "sensory experience"),
    ("feeling hungry", "AFF", "STA", "bodily state"),
    
    # ERG cases (AGENT) - must use DYN
    ("deliberately breaking a vase", "ERG", "DYN", "intentional action"),
    ("a person writing a letter", "ERG", "DYN", "volitional agent"),
    ("the chef cooking dinner", "ERG", "DYN", "agent performing action"),
    ("someone intentionally pushing a door", "ERG", "DYN", "deliberate force"),
    ("a student solving a problem", "ERG", "DYN", "cognitive agent"),
    
    # INS cases (INSTRUMENT) - must use DYN
    ("using a hammer to hit a nail", "INS", "DYN", "tool in active use"),
    ("the key that opens the door", "INS", "DYN", "instrument enabling action"),
    ("cutting with a knife", "INS", "DYN", "tool being wielded"),
    
    # ABS cases (PATIENT) - can use STA or DYN
    ("the vase that was broken", "ABS", "STA", "result state of patient"),
    ("the door being opened", "ABS", "DYN", "patient undergoing change"),
    
    # THM cases (CONTENT) - neutral, either function
    ("the topic of discussion", "THM", "STA", "static content"),
    ("what we are talking about", "THM", "STA", "thematic content"),
    
    # STM cases (STIMULUS) - trigger for experience
    ("the noise that startled me", "STM", "STA", "stimulus causing reaction"),
    ("the sight that frightened her", "STM", "STA", "stimulus for emotion"),
    
    # DAT cases (RECIPIENT) - typically DYN
    ("the person receiving a gift", "DAT", "DYN", "recipient of transfer"),
]


def run_experiment(pipeline: CopilotPipeline, num_cases: int = None) -> Dict:
    """
    Run experiment comparing RAG vs no-RAG accuracy
    
    Returns detailed results for analysis
    """
    cases = TEST_CASES[:num_cases] if num_cases else TEST_CASES
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(cases),
        "without_rag": {
            "valid": 0,
            "correct_case": 0,
            "correct_function": 0,
            "fully_correct": 0,
            "details": []
        },
        "with_rag": {
            "valid": 0,
            "correct_case": 0,
            "correct_function": 0,
            "fully_correct": 0,
            "details": []
        }
    }
    
    for i, (english, expected_case, expected_func, explanation) in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] Testing: '{english[:40]}...'")
        print(f"         Expected: {expected_case} + {expected_func}")
        
        # Test WITHOUT RAG
        suggestion, valid, msg = pipeline.suggest_without_rag(english)
        got_case = suggestion.get("case", "???")
        got_func = suggestion.get("function", "???")
        
        case_correct = (got_case == expected_case)
        func_correct = (got_func == expected_func)
        fully_correct = case_correct and func_correct
        
        results["without_rag"]["valid"] += int(valid)
        results["without_rag"]["correct_case"] += int(case_correct)
        results["without_rag"]["correct_function"] += int(func_correct)
        results["without_rag"]["fully_correct"] += int(fully_correct)
        results["without_rag"]["details"].append({
            "input": english,
            "expected": {"case": expected_case, "function": expected_func},
            "got": {"case": got_case, "function": got_func},
            "valid": valid,
            "case_correct": case_correct,
            "func_correct": func_correct,
            "reasoning": suggestion.get("reasoning", "")
        })
        
        status = "✅" if fully_correct else ("⚠️" if valid else "❌")
        print(f"         No RAG:  {got_case} + {got_func} {status}")
        
        # Test WITH RAG
        suggestion, valid, msg, attempts = pipeline.suggest_with_rag(english, max_retries=1)
        got_case = suggestion.get("case", "???")
        got_func = suggestion.get("function", "???")
        
        case_correct = (got_case == expected_case)
        func_correct = (got_func == expected_func)
        fully_correct = case_correct and func_correct
        
        results["with_rag"]["valid"] += int(valid)
        results["with_rag"]["correct_case"] += int(case_correct)
        results["with_rag"]["correct_function"] += int(func_correct)
        results["with_rag"]["fully_correct"] += int(fully_correct)
        results["with_rag"]["details"].append({
            "input": english,
            "expected": {"case": expected_case, "function": expected_func},
            "got": {"case": got_case, "function": got_func},
            "valid": valid,
            "case_correct": case_correct,
            "func_correct": func_correct,
            "attempts": attempts,
            "reasoning": suggestion.get("reasoning", "")
        })
        
        status = "✅" if fully_correct else ("⚠️" if valid else "❌")
        print(f"         With RAG: {got_case} + {got_func} {status} (attempts: {attempts})")
    
    return results


def print_summary(results: Dict):
    """Print experiment summary"""
    total = results["total_cases"]
    
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    
    print(f"\nTotal test cases: {total}")
    
    print("\n--- WITHOUT RAG (baseline) ---")
    no_rag = results["without_rag"]
    print(f"  Valid outputs:     {no_rag['valid']}/{total} ({100*no_rag['valid']/total:.1f}%)")
    print(f"  Correct case:      {no_rag['correct_case']}/{total} ({100*no_rag['correct_case']/total:.1f}%)")
    print(f"  Correct function:  {no_rag['correct_function']}/{total} ({100*no_rag['correct_function']/total:.1f}%)")
    print(f"  Fully correct:     {no_rag['fully_correct']}/{total} ({100*no_rag['fully_correct']/total:.1f}%)")
    
    print("\n--- WITH RAG ---")
    with_rag = results["with_rag"]
    print(f"  Valid outputs:     {with_rag['valid']}/{total} ({100*with_rag['valid']/total:.1f}%)")
    print(f"  Correct case:      {with_rag['correct_case']}/{total} ({100*with_rag['correct_case']/total:.1f}%)")
    print(f"  Correct function:  {with_rag['correct_function']}/{total} ({100*with_rag['correct_function']/total:.1f}%)")
    print(f"  Fully correct:     {with_rag['fully_correct']}/{total} ({100*with_rag['fully_correct']/total:.1f}%)")
    
    # Improvement
    improvement = with_rag['fully_correct'] - no_rag['fully_correct']
    if no_rag['fully_correct'] > 0:
        pct_improvement = 100 * improvement / no_rag['fully_correct']
    else:
        pct_improvement = float('inf') if improvement > 0 else 0
    
    print(f"\n📈 RAG IMPROVEMENT: +{improvement} cases ({pct_improvement:.1f}% relative improvement)")
    
    print("\n" + "=" * 70)


def save_results(results: Dict, filename: str = "experiment_results.json"):
    """Save full results to JSON"""
    output_path = Path(filename)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    # Allow specifying number of test cases
    num_cases = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    grammar_file = Path("../data/grammar_chunks.json")
    if not grammar_file.exists():
        grammar_file = Path("data/grammar_chunks.json")
    
    print("=" * 70)
    print("ITHKUIL VALIDATOR EXPERIMENT")
    print("Comparing LLM accuracy: Without RAG vs With RAG")
    print("=" * 70)
    
    pipeline = CopilotPipeline(grammar_file)
    
    results = run_experiment(pipeline, num_cases)
    print_summary(results)
    save_results(results)