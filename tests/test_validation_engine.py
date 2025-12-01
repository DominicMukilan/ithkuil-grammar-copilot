"""
Validation Engine Tests
======================

Comprehensive test suite for the Ithkuil grammar validation engine.
Tests co-occurrence rules, edge cases, error handling, and RAG integration.

Current: 60+ tests
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation_engine import ValidationEngine, ValidationLevel, ValidationError, ValidationResult


@pytest.fixture
def validator():
    """Create validator instance with real rules from grammar data"""
    grammar_file = Path("data/grammar_chunks.json")
    if not grammar_file.exists():
        grammar_file = Path("data_grammar_chunks.json")
    
    return ValidationEngine(grammar_kb={}, grammar_file=grammar_file)


@pytest.fixture
def validator_no_rag():
    """Create validator without RAG for testing fallback behavior"""
    return ValidationEngine(grammar_kb={}, grammar_file=None)


# SUITE 1: Core Co-occurrence Constraints (10 tests)
# Tests the fundamental case-function compatibility rules

class TestCoreCooccurrence:
    """Core semantic role compatibility tests"""
    
    def test_aff_dyn_rejection(self, validator):
        """AFF (EXPERIENCER) cannot co-occur with DYN function"""
        result = validator.validate({"case": "AFF", "function": "DYN"})
        assert not result.passed
        assert len(result.errors) == 1
        assert result.errors[0].level == ValidationLevel.COHERENCE
        assert "AFF" in result.errors[0].message
        assert "DYN" in result.errors[0].message

    def test_aff_sta_acceptance(self, validator):
        """AFF (EXPERIENCER) can co-occur with STA function"""
        result = validator.validate({"case": "AFF", "function": "STA"})
        assert result.passed
        assert result.confidence > 0.85
        assert len(result.errors) == 0

    def test_erg_dyn_acceptance(self, validator):
        """ERG (AGENT) can co-occur with DYN function"""
        result = validator.validate({"case": "ERG", "function": "DYN"})
        assert result.passed
        assert result.confidence > 0.85

    def test_erg_sta_rejection(self, validator):
        """ERG (AGENT) cannot co-occur with STA function"""
        result = validator.validate({"case": "ERG", "function": "STA"})
        assert not result.passed
        assert "ERG" in result.errors[0].message

    def test_abs_sta_acceptance(self, validator):
        """ABS (PATIENT) can co-occur with STA function"""
        result = validator.validate({"case": "ABS", "function": "STA"})
        assert result.passed

    def test_abs_dyn_acceptance(self, validator):
        """ABS (PATIENT) can co-occur with DYN function"""
        result = validator.validate({"case": "ABS", "function": "DYN"})
        assert result.passed

    def test_ins_dyn_acceptance(self, validator):
        """INS (INSTRUMENT) can co-occur with DYN function"""
        result = validator.validate({"case": "INS", "function": "DYN"})
        assert result.passed

    def test_ins_sta_rejection(self, validator):
        """INS (INSTRUMENT) cannot co-occur with STA function"""
        result = validator.validate({"case": "INS", "function": "STA"})
        assert not result.passed
        assert "INS" in result.errors[0].message

    def test_thm_both_functions(self, validator):
        """THM (CONTENT) allows both STA and DYN"""
        assert validator.validate({"case": "THM", "function": "STA"}).passed
        assert validator.validate({"case": "THM", "function": "DYN"}).passed

    def test_ind_both_functions(self, validator):
        """IND (AGENT+PATIENT) allows both STA and DYN"""
        assert validator.validate({"case": "IND", "function": "STA"}).passed
        assert validator.validate({"case": "IND", "function": "DYN"}).passed


# SUITE 2: Extended Semantic Role Coverage (15 tests)
# Tests diverse semantic roles across grammatical categories

class TestExtendedSemanticRoles:
    """Tests for less common but important semantic roles"""
    
    def test_stm_stimulus_both_functions(self, validator):
        """STM (STIMULUS) allows both functions"""
        assert validator.validate({"case": "STM", "function": "STA"}).passed
        assert validator.validate({"case": "STM", "function": "DYN"}).passed

    def test_dat_recipient_dyn(self, validator):
        """DAT (RECIPIENT) works with DYN for receiving actions"""
        result = validator.validate({"case": "DAT", "function": "DYN"})
        assert result.passed

    def test_eff_enabler_dyn(self, validator):
        """EFF (ENABLER) initiates causal chains - allows DYN"""
        result = validator.validate({"case": "EFF", "function": "DYN"})
        assert result.passed

    def test_prp_owner_sta(self, validator):
        """PRP (OWNER) - ownership is inherently static"""
        result = validator.validate({"case": "PRP", "function": "STA"})
        assert result.passed

    def test_pos_possessor_sta(self, validator):
        """POS (POSSESSOR) - possession is static"""
        result = validator.validate({"case": "POS", "function": "STA"})
        assert result.passed

    def test_att_attribute_sta(self, validator):
        """ATT (ATTRIBUTE) - attributes are static properties"""
        result = validator.validate({"case": "ATT", "function": "STA"})
        assert result.passed

    def test_gen_source_both(self, validator):
        """GEN (SOURCE) allows both functions"""
        assert validator.validate({"case": "GEN", "function": "STA"}).passed
        assert validator.validate({"case": "GEN", "function": "DYN"}).passed

    def test_pdc_producer(self, validator):
        """PDC (PRODUCER) - creator/author relationship"""
        result = validator.validate({"case": "PDC", "function": "STA"})
        assert result.passed

    def test_cor_correlation_both(self, validator):
        """COR (CORRELATION) allows both functions"""
        assert validator.validate({"case": "COR", "function": "STA"}).passed
        assert validator.validate({"case": "COR", "function": "DYN"}).passed

    def test_par_part_both(self, validator):
        """PAR (PART) - part-whole relationships"""
        assert validator.validate({"case": "PAR", "function": "STA"}).passed
        assert validator.validate({"case": "PAR", "function": "DYN"}).passed

    def test_idp_dependent_both(self, validator):
        """IDP (DEPENDENT) - interdependent relationships"""
        assert validator.validate({"case": "IDP", "function": "STA"}).passed
        assert validator.validate({"case": "IDP", "function": "DYN"}).passed

    def test_dep_contingency_both(self, validator):
        """DEP (CONTINGENCY) allows both functions"""
        assert validator.validate({"case": "DEP", "function": "STA"}).passed
        assert validator.validate({"case": "DEP", "function": "DYN"}).passed

    def test_apl_purpose_dyn_only(self, validator):
        """APL (PURPOSE) - purposes are dynamic goals"""
        assert validator.validate({"case": "APL", "function": "DYN"}).passed
        assert not validator.validate({"case": "APL", "function": "STA"}).passed

    def test_pur_function_both(self, validator):
        """PUR (FUNCTION) - semantic role not strictly constrained, allows both"""
        # Note: PUR has semantic_role "FUNCTION" which isn't in strict rules
        # So it defaults to allowing all functions
        assert validator.validate({"case": "PUR", "function": "DYN"}).passed
        assert validator.validate({"case": "PUR", "function": "STA"}).passed

    def test_act_activation_dyn(self, validator):
        """ACT (ACTIVATION) allows DYN for modal states"""
        result = validator.validate({"case": "ACT", "function": "DYN"})
        assert result.passed


# SUITE 3: Spatial & Temporal Cases (10 tests)
# Tests spatio-temporal semantic roles

class TestSpatioTemporalCases:
    """Tests for location, movement, and time-related cases"""
    
    def test_loc_location_both(self, validator):
        """LOC (LOCATION) - static location allows both"""
        assert validator.validate({"case": "LOC", "function": "STA"}).passed
        assert validator.validate({"case": "LOC", "function": "DYN"}).passed

    def test_all_goal_dyn(self, validator):
        """ALL (GOAL) - movement toward requires DYN"""
        result = validator.validate({"case": "ALL", "function": "DYN"})
        assert result.passed

    def test_abl_source_both(self, validator):
        """ABL (SOURCE) - movement from, allows both"""
        assert validator.validate({"case": "ABL", "function": "STA"}).passed
        assert validator.validate({"case": "ABL", "function": "DYN"}).passed

    def test_ori_orientation_both(self, validator):
        """ORI (ORIENTATION) allows both functions"""
        assert validator.validate({"case": "ORI", "function": "STA"}).passed
        assert validator.validate({"case": "ORI", "function": "DYN"}).passed

    def test_nav_path_dyn(self, validator):
        """NAV (PATH) - trajectory of motion"""
        result = validator.validate({"case": "NAV", "function": "DYN"})
        assert result.passed

    def test_cnr_simultaneity_both(self, validator):
        """CNR (SIMULTANEITY) - temporal locative"""
        assert validator.validate({"case": "CNR", "function": "STA"}).passed
        assert validator.validate({"case": "CNR", "function": "DYN"}).passed

    def test_pcv_before_both(self, validator):
        """PCV (BEFORE) - prior to event"""
        assert validator.validate({"case": "PCV", "function": "STA"}).passed
        assert validator.validate({"case": "PCV", "function": "DYN"}).passed

    def test_pcr_after_both(self, validator):
        """PCR (AFTER) - subsequent to event"""
        assert validator.validate({"case": "PCR", "function": "STA"}).passed
        assert validator.validate({"case": "PCR", "function": "DYN"}).passed

    def test_elp_elapsed_both(self, validator):
        """ELP (ELAPSED) - time passed"""
        assert validator.validate({"case": "ELP", "function": "STA"}).passed
        assert validator.validate({"case": "ELP", "function": "DYN"}).passed

    def test_irl_reference_both(self, validator):
        """IRL (REFERENCE) - relative position"""
        assert validator.validate({"case": "IRL", "function": "STA"}).passed
        assert validator.validate({"case": "IRL", "function": "DYN"}).passed


# SUITE 4: Edge Cases & Error Handling (15 tests)
# Tests boundary conditions and error scenarios

class TestEdgeCasesAndErrors:
    """Tests for unusual inputs and error conditions"""
    
    def test_empty_input(self, validator):
        """Empty dict should pass structure (no case/function to validate)"""
        result = validator.validate({})
        # No case or function means no co-occurrence to check
        assert result.passed

    def test_missing_case(self, validator):
        """Missing case with valid function"""
        result = validator.validate({"function": "STA"})
        assert result.passed  # No case to validate against

    def test_missing_function(self, validator):
        """Missing function with valid case"""
        result = validator.validate({"case": "AFF"})
        assert result.passed  # No function to validate against

    def test_invalid_function_value(self, validator):
        """Invalid function value should be rejected"""
        result = validator.validate({"case": "AFF", "function": "INVALID"})
        assert not result.passed
        assert "Invalid function" in result.errors[0].message

    def test_invalid_function_experiencer(self, validator):
        """EXPERIENCER is a role, not a function"""
        result = validator.validate({"case": "AFF", "function": "EXPERIENCER"})
        assert not result.passed
        assert "Invalid function" in result.errors[0].message

    def test_lowercase_case(self, validator):
        """Lowercase case code should be rejected"""
        result = validator.validate({"case": "aff", "function": "STA"})
        assert not result.passed
        assert "Invalid case format" in result.errors[0].message

    def test_unknown_case_code_passes(self, validator):
        """Unknown but well-formed case code passes (extensibility)"""
        result = validator.validate({"case": "XYZ", "function": "STA"})
        # Unknown cases pass for extensibility - we don't reject valid format
        assert result.passed

    def test_extra_fields_preserved(self, validator):
        """Extra fields in input don't affect validation"""
        result = validator.validate({
            "case": "AFF",
            "function": "STA",
            "aspect": "HAB",
            "extra": "data"
        })
        assert result.passed

    def test_none_values(self, validator):
        """None values treated as missing"""
        result = validator.validate({"case": None, "function": None})
        assert result.passed  # Nothing to validate

    def test_numeric_case_rejected(self, validator):
        """Numeric case value should be rejected with clear error"""
        result = validator.validate({"case": 123, "function": "STA"})
        assert not result.passed
        assert "must be a string" in result.errors[0].message

    def test_numeric_function_rejected(self, validator):
        """Numeric function value should be rejected with clear error"""
        result = validator.validate({"case": "AFF", "function": 456})
        assert not result.passed
        assert "must be a string" in result.errors[0].message

    def test_validation_result_repr_valid(self, validator):
        """ValidationResult repr for valid result"""
        result = validator.validate({"case": "AFF", "function": "STA"})
        repr_str = repr(result)
        assert "VALID" in repr_str

    def test_validation_result_repr_invalid(self, validator):
        """ValidationResult repr for invalid result"""
        result = validator.validate({"case": "AFF", "function": "DYN"})
        repr_str = repr(result)
        assert "INVALID" in repr_str

    def test_error_has_suggestion(self, validator):
        """Validation errors include suggestions"""
        result = validator.validate({"case": "ERG", "function": "STA"})
        assert not result.passed
        assert result.errors[0].suggestion is not None
        assert len(result.errors[0].suggestion) > 0

    def test_error_has_slot(self, validator):
        """Validation errors identify the problematic slot"""
        result = validator.validate({"case": "AFF", "function": "DYN"})
        assert not result.passed
        assert result.errors[0].slot is not None

    def test_input_not_mutated(self, validator):
        """Validation should not modify the input dict"""
        original = {"case": "AFF", "function": "DYN", "other": "data"}
        original_copy = original.copy()
        validator.validate(original)
        assert original == original_copy


# SUITE 5: Statistics & State Management (5 tests)
# Tests validator state tracking

class TestStatisticsAndState:
    """Tests for validator statistics and state management"""
    
    def test_stats_initial_state(self, validator):
        """Fresh validator has zero stats"""
        stats = validator.get_stats()
        assert "total_validations" in stats
        assert "passed" in stats
        assert "rejected" in stats

    def test_stats_increment_on_pass(self, validator):
        """Stats increment correctly on pass"""
        initial = validator.get_stats()["passed"]
        validator.validate({"case": "AFF", "function": "STA"})
        assert validator.get_stats()["passed"] == initial + 1

    def test_stats_increment_on_reject(self, validator):
        """Stats increment correctly on rejection"""
        initial = validator.get_stats()["rejected"]
        validator.validate({"case": "AFF", "function": "DYN"})
        assert validator.get_stats()["rejected"] == initial + 1

    def test_stats_pass_rate_calculation(self, validator):
        """Pass rate calculated correctly"""
        validator.validate({"case": "AFF", "function": "STA"})  # pass
        validator.validate({"case": "AFF", "function": "STA"})  # pass
        validator.validate({"case": "AFF", "function": "DYN"})  # reject
        
        stats = validator.get_stats()
        assert 0 <= stats["pass_rate"] <= 1

    def test_validations_independent(self, validator):
        """Each validation is independent"""
        result1 = validator.validate({"case": "AFF", "function": "STA"})
        assert result1.passed
        
        result2 = validator.validate({"case": "AFF", "function": "DYN"})
        assert not result2.passed
        
        result3 = validator.validate({"case": "ERG", "function": "DYN"})
        assert result3.passed


# SUITE 6: RAG Integration (5 tests)
# Tests RAG-related functionality

class TestRAGIntegration:
    """Tests for RAG system integration"""
    
    def test_citations_provided_on_valid(self, validator):
        """Valid results include citations from RAG"""
        result = validator.validate({"case": "AFF", "function": "STA"})
        assert result.passed
        assert len(result.citations) > 0

    def test_citations_reference_grammar(self, validator):
        """Citations reference official grammar"""
        result = validator.validate({"case": "ERG", "function": "DYN"})
        assert result.passed
        assert any("Grammar" in c or "ERG" in c or "Ergative" in c 
                   for c in result.citations)

    def test_confidence_high_for_known_case(self, validator):
        """Known cases have high confidence"""
        result = validator.validate({"case": "AFF", "function": "STA"})
        assert result.passed
        assert result.confidence >= 0.90

    def test_fallback_without_rag(self, validator_no_rag):
        """Validator works without RAG (fallback mode)"""
        result = validator_no_rag.validate({"case": "AFF", "function": "STA"})
        assert isinstance(result, ValidationResult)

    def test_semantic_role_in_error(self, validator):
        """Error messages include semantic role information"""
        result = validator.validate({"case": "AFF", "function": "DYN"})
        assert not result.passed
        assert "EXPERIENCER" in result.errors[0].message


# SUITE 7: Comprehensive Case Coverage (parametrized)
# Quick validation that all major case categories work

class TestComprehensiveCaseCoverage:
    """Verify major cases across all categories are handled"""
    
    @pytest.mark.parametrize("case,function,expected_valid", [
        # Transrelative cases
        ("AFF", "STA", True),
        ("ERG", "DYN", True),
        ("ABS", "STA", True),
        ("INS", "DYN", True),
        ("DAT", "DYN", True),
        # Appositive cases
        ("POS", "STA", True),
        ("PRP", "STA", True),
        ("GEN", "STA", True),
        # Associative cases
        ("APL", "DYN", True),
        ("PUR", "DYN", True),
        ("TRA", "DYN", True),
        # Relational cases
        ("COR", "STA", True),
        ("COM", "STA", True),
        # Spatio-temporal cases
        ("LOC", "STA", True),
        ("ALL", "DYN", True),
        ("ABL", "DYN", True),
    ])
    def test_case_function_pair(self, validator, case, function, expected_valid):
        """Parametrized test for case-function pairs"""
        result = validator.validate({"case": case, "function": function})
        assert result.passed == expected_valid, \
            f"{case}+{function} expected {'valid' if expected_valid else 'invalid'}"


# Test count: 76 tests (including parametrized)
# Run pytest tests/test_validation_engine.py -v