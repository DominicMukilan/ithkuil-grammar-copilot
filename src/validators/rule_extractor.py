"""
Rule Extractor - Extract Validation Rules from Grammar Data
===========================================================

Converts raw grammar JSON into structured validation rules.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass


@dataclass
class CaseConstraints:
    """Constraints for a specific grammatical case"""
    case_name: str
    case_abbrev: str
    semantic_role: str
    description: str
    allowed_functions: Set[str]
    forbidden_functions: Set[str]
    why_not_alternatives: Dict[str, str]
    common_mistakes: List[str]
    
    def allows_function(self, function: str) -> bool:
        """Check if this case allows a specific function"""
        if self.forbidden_functions and function in self.forbidden_functions:
            return False
        if self.allowed_functions and function not in self.allowed_functions:
            return False
        return True


class RuleExtractor:
    """
    Extract validation rules from grammar data files
    """
    
    # Semantic role → function compatibility (based on Ithkuil semantics)
    ROLE_FUNCTION_RULES = {
        # Core roles
        "EXPERIENCER": {"allowed": {"STA"}, "forbidden": {"DYN", "MNF"}},  # AFF - unwilled experiences
        "AGENT": {"allowed": {"DYN", "MNF"}, "forbidden": {"STA"}},        # ERG - willed actions
        "PATIENT": {"allowed": {"STA", "DYN"}, "forbidden": set()},        # ABS - undergoes change
        "AGENT+PATIENT": {"allowed": {"STA", "DYN"}, "forbidden": set()},  # IND - both roles
        
        # Instrument & tools
        "INSTRUMENT": {"allowed": {"DYN"}, "forbidden": {"STA"}},          # INS - active use
        
        # Content & theme
        "CONTENT": {"allowed": {"STA", "DYN", "MNF"}, "forbidden": set()}, # THM - neutral participant
        "STIMULUS": {"allowed": {"STA", "DYN"}, "forbidden": set()},       # STM - trigger
        
        # Dynamic roles (inherently action-oriented)
        "GOAL": {"allowed": {"DYN", "MNF"}, "forbidden": {"STA"}},         # ALL - target of motion/action
        "PURPOSE": {"allowed": {"DYN", "MNF"}, "forbidden": {"STA"}},      # APL - intended outcome
        "ENABLER": {"allowed": {"DYN", "MNF"}, "forbidden": set()},        # EFF - makes action possible
        "ACTIVATION": {"allowed": {"DYN", "MNF"}, "forbidden": set()},     # ACT - initiates action
        "RECIPIENT": {"allowed": {"DYN"}, "forbidden": set()},             # DAT - receives something
        
        # Static roles (inherently state-oriented)
        "ATTRIBUTE": {"allowed": {"STA"}, "forbidden": set()},             # ATT - properties
        "POSSESSOR": {"allowed": {"STA"}, "forbidden": set()},             # POS - has something
        "OWNER": {"allowed": {"STA"}, "forbidden": set()},                 # PRP - owns something
        
        # Spatial & temporal (can be static or dynamic)
        "LOCATION": {"allowed": {"STA", "DYN"}, "forbidden": set()},       # LOC - place
        "ORIENTATION": {"allowed": {"STA", "DYN"}, "forbidden": set()},    # ORI - direction
        "SOURCE": {"allowed": {"STA", "DYN"}, "forbidden": set()},         # GEN/ABL - origin
        
        # Relational (typically static)
        "PART": {"allowed": {"STA", "DYN"}, "forbidden": set()},           # PAR - part-whole
        "CORRELATION": {"allowed": {"STA", "DYN"}, "forbidden": set()},    # COR - relationship
        "DEPENDENT": {"allowed": {"STA", "DYN"}, "forbidden": set()},      # IDP - depends on
        "CONTINGENCY": {"allowed": {"STA", "DYN"}, "forbidden": set()},    # DEP - conditional
    }
    
    def __init__(self, grammar_file: Path):
        """
        Initialize rule extractor
        
        Args:
            grammar_file: Path to grammar_chunks.json
        """
        self.grammar_file = grammar_file
        self.raw_data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and parse grammar JSON"""
        with open(self.grammar_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_case_constraints(self) -> Dict[str, CaseConstraints]:
        """
        Extract case-function compatibility rules from grammar data
        
        Returns:
            Dictionary mapping case abbreviations to their constraints
        """
        constraints = {}
        
        # Extract all cases from the data
        cases = [item for item in self.raw_data if item.get('type') == 'case']
        
        for case_data in cases:
            code = case_data.get('code')
            semantic_role = case_data.get('semantic_role', 'UNKNOWN')
            
            # Get function rules based on semantic role
            role_rules = self.ROLE_FUNCTION_RULES.get(
                semantic_role,
                {"allowed": {"STA", "DYN", "MNF"}, "forbidden": set()}  # Default: allow all
            )
            
            constraints[code] = CaseConstraints(
                case_name=case_data.get('name', code),
                case_abbrev=code,
                semantic_role=semantic_role,
                description=case_data.get('description', ''),
                allowed_functions=role_rules["allowed"].copy(),
                forbidden_functions=role_rules["forbidden"].copy(),
                why_not_alternatives=case_data.get('why_not_alternatives', {}),
                common_mistakes=case_data.get('common_mistakes', [])
            )
        
        return constraints
    
    def validate_rules(self) -> bool:
        """
        Validate that extracted rules are consistent
        
        Returns:
            True if rules are valid
        """
        constraints = self.extract_case_constraints()
        
        # Check for conflicts
        for case_abbrev, constraint in constraints.items():
            # Can't have overlapping allowed/forbidden
            overlap = constraint.allowed_functions & constraint.forbidden_functions
            if overlap:
                print(f"WARNING: {case_abbrev} has conflicting rules for {overlap}")
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about extracted rules"""
        constraints = self.extract_case_constraints()
        
        # Count by strictness
        strict_cases = sum(1 for c in constraints.values() if c.forbidden_functions)
        permissive_cases = sum(1 for c in constraints.values() if not c.forbidden_functions)
        
        # Count by function
        function_counts = {
            "STA": sum(1 for c in constraints.values() if "STA" in c.allowed_functions),
            "DYN": sum(1 for c in constraints.values() if "DYN" in c.allowed_functions),
            "MNF": sum(1 for c in constraints.values() if "MNF" in c.allowed_functions),
        }
        
        return {
            "total_cases": len(constraints),
            "strict_cases": strict_cases,
            "permissive_cases": permissive_cases,
            "cases_by_function": function_counts,
            "semantic_roles_covered": len(set(c.semantic_role for c in constraints.values()))
        }


# Quick test
if __name__ == "__main__":
    grammar_file = Path("data_grammar_chunks.json")
    if not grammar_file.exists():
        grammar_file = Path("data/grammar_chunks.json")
    
    extractor = RuleExtractor(grammar_file)
    
    print("📊 Extracting rules from grammar data...")
    constraints = extractor.extract_case_constraints()
    
    print(f"\n✅ Extracted {len(constraints)} case rules\n")
    
    # Show interesting cases including the fixed ones
    print("Sample Case Rules (based on semantic roles):\n")
    sample_cases = ["AFF", "ERG", "ABS", "INS", "THM", "IND", "PRP"]
    for case in sample_cases:
        if case in constraints:
            c = constraints[case]
            print(f"{case} - {c.case_name} ({c.semantic_role}):")
            print(f"  ✓ Allowed: {', '.join(sorted(c.allowed_functions))}")
            if c.forbidden_functions:
                print(f"  ✗ Forbidden: {', '.join(sorted(c.forbidden_functions))}")
            print(f"  Description: {c.description[:80]}...")
            print()
    
    print("\n🔍 Validating rule consistency...")
    if extractor.validate_rules():
        print("✅ All rules are consistent!\n")
    else:
        print("❌ Rule conflicts detected!\n")
    
    print("📈 Statistics:")
    stats = extractor.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")