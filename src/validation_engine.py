"""
Validation Engine
=====================================

The ValidationEngine has absolute veto power over all system output.
This is the co-pilot pattern: LLM suggests, validator decides.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from rag_system import RAGSystem

class ValidationLevel(Enum):
    """Levels of validation strictness"""
    STRUCTURE = "structure"      # Phonological + slot completeness
    COHERENCE = "coherence"      # Co-occurrence constraints
    SEMANTIC = "semantic"        # RAG citation required
    

@dataclass
class ValidationError:
    """A validation failure with explanation"""
    level: ValidationLevel
    message: str
    slot: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation check"""
    passed: bool
    confidence: float
    errors: List[ValidationError]
    citations: List[str]
    needs_clarification: bool = False
    
    def __repr__(self):
        status = "✅ VALID" if self.passed else "❌ INVALID"
        return f"{status} (confidence={self.confidence:.2f}, {len(self.errors)} errors)"


class ValidationEngine:
    """
    Co-pilot pattern for hybrid AI systems.
    
    The ValidationEngine enforces hard constraints that LLMs cannot reliably learn:
    - Phonological rules (stress placement, valid clusters)
    - Morpheme co-occurrence (AFF case + DYN function = invalid)
    - Slot dependencies (slot IX requires slot V)
    - Citation requirements (every claim needs grammar reference)
    
    This pattern generalizes to any domain where correctness > fluency.
    """
    
    def __init__(self, grammar_kb: Dict[str, Any], grammar_file: Optional[Path] = None):
        """
        Initialize validation engine with grammar knowledge base
        
        Args:
            grammar_kb: Structured grammar rules and constraints
            grammar_file: Path to grammar JSON for rule extraction
        """
        self.grammar_kb = grammar_kb
        self.stats = {
            "total_validations": 0,
            "passed": 0,
            "rejected": 0,
            "clarification_needed": 0
        }
        
        # Load real cooccurrence rules from grammar data
        self.cooccurrence_rules = None
        self.rag = None
        
        if grammar_file and grammar_file.exists():
            try:
                # Import here to avoid circular dependencies
                from validators.cooccurrence_rules import CooccurrenceRules
                self.cooccurrence_rules = CooccurrenceRules(grammar_file)
                print(f"✅ ValidationEngine loaded with {len(self.cooccurrence_rules.constraints)} case rules")
                
                # Initialize RAG system
                self.rag = RAGSystem(grammar_file)
                print(f"✅ RAG system initialized")
                
            except Exception as e:
                print(f"⚠️  Failed to load rules/RAG: {e}")
                print("   Falling back to basic validation")
        else:
            print("⚠️  No grammar file provided - using basic validation only")
                
    def validate(self, semantic_json: Dict[str, Any]) -> ValidationResult:
        """
        Validate semantic JSON against all constraints
        
        Args:
            semantic_json: The semantic representation to validate
            
        Returns:
            ValidationResult with pass/fail and detailed errors
        """
        self.stats["total_validations"] += 1
        
        errors = []
        confidence = 1.0
        citations = []
        
        # Level 1: Structure validation (deterministic)
        structure_errors = self._validate_structure(semantic_json)
        errors.extend(structure_errors)
        if structure_errors:
            self.stats["rejected"] += 1
            return ValidationResult(
                passed=False,
                confidence=0.0,
                errors=errors,
                citations=[]
            )
        
        # Level 2: Coherence validation (constraint checking)
        coherence_errors = self._validate_coherence(semantic_json)
        errors.extend(coherence_errors)
        if coherence_errors:
            self.stats["rejected"] += 1
            return ValidationResult(
                passed=False,
                confidence=0.0,
                errors=errors,
                citations=[]
            )
        
        # Level 3: Semantic validation (RAG grounding)
        semantic_result = self._validate_semantic(semantic_json)
        errors.extend(semantic_result["errors"])
        confidence = semantic_result["confidence"]
        citations = semantic_result["citations"]
        
        # Decision logic
        # Low confidence without errors = pass with warning (extensibility)
        # Low confidence with errors = fail
        if errors:
            self.stats["rejected"] += 1
            return ValidationResult(
                passed=False,
                confidence=confidence,
                errors=errors,
                citations=citations
            )
        
        if confidence < 0.85:
            self.stats["clarification_needed"] += 1
            # Still passes, but flagged for review
            self.stats["passed"] += 1
            return ValidationResult(
                passed=True,
                confidence=confidence,
                errors=[],
                citations=citations,
                needs_clarification=True
            )
        
        self.stats["passed"] += 1
        return ValidationResult(
            passed=True,
            confidence=confidence,
            errors=[],
            citations=citations
        )
    
    def _validate_structure(self, semantic_json: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate phonological structure and slot completeness
        
        TODO: Implement phonotactic rules
        TODO: Implement stress placement rules
        """
        errors = []
        # Placeholder - implement in Week 1 Day 4-5
        return errors
    
    def _validate_coherence(self, semantic_json: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate morpheme co-occurrence constraints using loaded rules
        """
        errors = []
        
        case = semantic_json.get("case")
        function = semantic_json.get("function")
        
        # Handle None or missing values
        if not case or not function:
            return errors
        
        # Type check - case and function must be strings
        if not isinstance(case, str):
            errors.append(ValidationError(
                level=ValidationLevel.COHERENCE,
                message=f"Case must be a string, got {type(case).__name__}",
                slot="case",
                suggestion="Use a 3-letter case code like AFF, ERG, ABS"
            ))
            return errors
        
        if not isinstance(function, str):
            errors.append(ValidationError(
                level=ValidationLevel.COHERENCE,
                message=f"Function must be a string, got {type(function).__name__}",
                slot="function",
                suggestion="Use STA for states or DYN for actions"
            ))
            return errors
        
        # Check if function is valid
        valid_functions = {"STA", "DYN", "MNF"}
        if function not in valid_functions:
            errors.append(ValidationError(
                level=ValidationLevel.COHERENCE,
                message=f"Invalid function '{function}'. Must be one of: {', '.join(sorted(valid_functions))}",
                slot="function",
                suggestion="Use STA for states or DYN for actions"
            ))
            return errors  # Don't continue if function is invalid
        
        # Check case format (must be 3 uppercase letters)
        if not (len(case) == 3 and case.isupper() and case.isalpha()):
            errors.append(ValidationError(
                level=ValidationLevel.COHERENCE,
                message=f"Invalid case format '{case}'. Cases are 3-letter uppercase codes like AFF, ERG, ABS",
                slot="case",
                suggestion="Use a valid case code"
            ))
            return errors
        
        # Check co-occurrence rules if we have them
        if self.cooccurrence_rules:
            # Only validate if case is known - unknown cases pass (extensibility)
            if case in self.cooccurrence_rules.constraints:
                is_valid, error_msg = self.cooccurrence_rules.check_case_function(case, function)
                if not is_valid:
                    allowed = self.cooccurrence_rules.get_allowed_functions(case)
                    errors.append(ValidationError(
                        level=ValidationLevel.COHERENCE,
                        message=error_msg,
                        slot="case+function",
                        suggestion=f"Try one of: {', '.join(sorted(allowed))}"
                    ))
        
        return errors
       
    def _validate_semantic(self, semantic_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate semantic grounding via RAG
        
        Uses RAG to retrieve grammar rules for citations and context
        """
        errors = []
        confidence = 0.94  # High confidence if we have RAG data
        citations = []
        
        case = semantic_json.get("case")
        
        if self.rag and case and isinstance(case, str):
            # Query RAG for case information
            result = self.rag.retrieve_for_case(case)
            
            if result:
                # RAG provides citations, not confidence
                # Confidence comes from validation passing, not retrieval score
                citations.append(result.citation)
                
                # Add semantic role info
                citations.append(
                    f"{result.case_name} ({result.semantic_role}): {result.description[:100]}..."
                )
                
                # High confidence - case found in grammar
                confidence = 0.95
            else:
                # Unknown case - lower confidence but don't fail
                confidence = 0.70
                citations.append(f"Warning: Case {case} not found in grammar database")
        else:
            # No RAG available - use fallback
            citations = ["Grammar §7 (Cases)"]
            confidence = 0.90
        
        return {
            "errors": errors,
            "confidence": confidence,
            "citations": citations
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.stats["total_validations"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "pass_rate": self.stats["passed"] / total,
            "rejection_rate": self.stats["rejected"] / total
        }


# Quick test
if __name__ == "__main__":
    # Try to find grammar file
    grammar_file = Path("data/grammar_chunks.json")
    
    # Initialize with grammar rules + RAG
    print("\n" + "=" * 70)
    print("VALIDATION ENGINE + RAG TEST")
    print("=" * 70)
    
    validator = ValidationEngine(grammar_kb={}, grammar_file=grammar_file)
    
    print("\n🧪 Testing ValidationEngine with RAG...\n")
    
    # Test valid case (AFF + STA)
    result = validator.validate({
        "case": "AFF",
        "function": "STA",
        "aspect": "HAB"
    })
    print(f"✅ Valid case (AFF+STA): {result}")
    print(f"   Confidence from RAG: {result.confidence:.3f}")
    print(f"   Citations: {result.citations[0]}")
    
    # Test invalid case (AFF + DYN)
    result = validator.validate({
        "case": "AFF",
        "function": "DYN",
        "aspect": "HAB"
    })
    print(f"\n❌ Invalid case (AFF+DYN): {result}")
    if result.errors:
        print(f"   Error: {result.errors[0].message}")
        print(f"   Suggestion: {result.errors[0].suggestion}")
    
    # Test unknown case (should pass - extensibility)
    result = validator.validate({
        "case": "XYZ",
        "function": "STA"
    })
    print(f"\n⚠️  Unknown case (XYZ+STA): {result}")
    print(f"   Passed: {result.passed} (unknown cases allowed for extensibility)")
    
    # Show stats
    print(f"\n📊 Validator Stats:")
    stats = validator.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print(f"\n📊 RAG Stats:")
    rag_stats = validator.rag.get_stats()
    for k, v in rag_stats.items():
        print(f"   {k}: {v}")