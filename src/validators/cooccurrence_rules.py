"""
Cooccurrence Rules - Smart Validation Logic
==========================================

Loads and applies grammatical constraints from extracted rules.
"""

from typing import Dict, Optional, Set, Tuple, List
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .rule_extractor import RuleExtractor, CaseConstraints
except ImportError:
    # Running as script
    from rule_extractor import RuleExtractor, CaseConstraints

class CooccurrenceRules:
    """
    Manages case-function compatibility rules
    
    This is where linguistic knowledge meets code.
    Rules are extracted from grammar data based on semantic roles.
    """
    
    def __init__(self, grammar_file: Path):
        """
        Load rules from grammar data
        
        Args:
            grammar_file: Path to grammar data JSON
        """
        self.grammar_file = grammar_file
        self.extractor = RuleExtractor(grammar_file)
        self.constraints = self.extractor.extract_case_constraints()
        
        # Validate on load
        if not self.extractor.validate_rules():
            raise ValueError("Inconsistent rules detected in grammar data")
        
        print(f"✅ Loaded {len(self.constraints)} case rules from {grammar_file.name}")
    
    def check_case_function(self, case: str, function: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a case can co-occur with a function
        
        Args:
            case: Case abbreviation (e.g., "AFF")
            function: Function abbreviation (e.g., "STA")
            
        Returns:
            (is_valid, error_message)
        """
        if case not in self.constraints:
            return (True, None)  # Unknown case - fail open for extensibility
        
        constraint = self.constraints[case]
        
        if not constraint.allows_function(function):
            allowed = constraint.allowed_functions
            forbidden = constraint.forbidden_functions
            
            msg = (f"{case} ({constraint.case_name}) cannot co-occur with {function} function. "
                   f"Semantic role: {constraint.semantic_role}. ")
            
            if allowed:
                msg += f"Allowed: {', '.join(sorted(allowed))}. "
            if forbidden:
                msg += f"Forbidden: {', '.join(sorted(forbidden))}."
            
            return (False, msg)
        
        return (True, None)
    
    def get_allowed_functions(self, case: str) -> Set[str]:
        """Get all functions allowed for a case"""
        if case not in self.constraints:
            return {"STA", "DYN", "MNF"}  # Default: allow all
        return self.constraints[case].allowed_functions
    
    def get_case_description(self, case: str) -> str:
        """Get semantic description of a case"""
        if case not in self.constraints:
            return f"Unknown case: {case}"
        c = self.constraints[case]
        return f"{c.case_name} ({c.semantic_role}): {c.description[:100]}..."
    
    def get_why_not_alternative(self, case: str, alternative: str) -> Optional[str]:
        """Get explanation for why an alternative case doesn't work"""
        if case not in self.constraints:
            return None
        return self.constraints[case].why_not_alternatives.get(alternative)
    
    def get_common_mistakes(self, case: str) -> List[str]:
        """Get common mistakes for a case"""
        if case not in self.constraints:
            return []
        return self.constraints[case].common_mistakes
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded rules"""
        return self.extractor.get_stats()


# Quick test
if __name__ == "__main__":
    grammar_file = Path("data_grammar_chunks.json")
    if not grammar_file.exists():
        grammar_file = Path("data/grammar_chunks.json")
    
    rules = CooccurrenceRules(grammar_file)
    
    print("\n🧪 Testing case-function rules...\n")
    
    # Test cases based on semantic roles
    tests = [
        ("AFF", "STA", True),   # EXPERIENCER + Static = OK
        ("AFF", "DYN", False),  # EXPERIENCER + Dynamic = NO
        ("ERG", "DYN", True),   # AGENT + Dynamic = OK
        ("ERG", "STA", False),  # AGENT + Static = NO
        ("ABS", "STA", True),   # PATIENT + Static = OK
        ("ABS", "DYN", True),   # PATIENT + Dynamic = OK
        ("INS", "DYN", True),   # INSTRUMENT + Dynamic = OK
        ("INS", "STA", False),  # INSTRUMENT + Static = NO
    ]
    
    passed = 0
    for case, function, expected_valid in tests:
        is_valid, error = rules.check_case_function(case, function)
        status = "✅" if is_valid == expected_valid else "❌"
        result = "PASS" if is_valid else "FAIL"
        print(f"{status} {case} + {function}: {result}")
        if error:
            print(f"   → {error}")
        if is_valid == expected_valid:
            passed += 1
    
    print(f"\n📊 Tests: {passed}/{len(tests)} passed")
    print("\n📈 Rule Stats:", rules.get_stats())