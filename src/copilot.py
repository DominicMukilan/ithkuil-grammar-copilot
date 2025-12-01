# src/copilot.py
"""
Co-Pilot Pipeline
============================

Connects: LLM → RAG → Validator
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from llm_client import LLMClient
from rag_system import RAGSystem
from validation_engine import ValidationEngine


class CopilotPipeline:
    """
    The actual co-pilot pattern:
    1. User provides English description
    2. RAG retrieves relevant grammar rules
    3. LLM suggests case+function using RAG context
    4. Validator accepts or rejects
    5. If rejected, LLM tries again with feedback
    """
    
    def __init__(self, grammar_file: Path):
        """Initialize all components"""
        print("🔄 Initializing Co-Pilot Pipeline...")
        
        self.grammar_file = grammar_file
        
        # Initialize components
        self.llm = LLMClient()
        self.rag = RAGSystem(grammar_file)
        self.validator = ValidationEngine(grammar_kb={}, grammar_file=grammar_file)
        
        print("✅ Pipeline ready")
    
    def _build_prompt_without_rag(self, user_input: str) -> str:
        """Build prompt WITHOUT RAG context (baseline)"""
        return f"""You are an Ithkuil IV grammar expert.

TASK: Given an English description, select the correct CASE and FUNCTION.

VALID CASES: AFF, ERG, ABS, INS, THM, DAT, LOC, ALL, ABL, etc. (68 total)
VALID FUNCTIONS: STA (static/states) or DYN (dynamic/actions)

English description: "{user_input}"

Respond with ONLY valid JSON:
{{"case": "THREE_LETTER_CODE", "function": "STA_or_DYN", "reasoning": "why"}}
"""
    
    def _build_prompt_with_rag(self, user_input: str, rag_context: str) -> str:
        """Build prompt WITH RAG context"""
        return f"""You are an Ithkuil IV grammar expert.

TASK: Given an English description, select the correct CASE and FUNCTION.

VALID CASES (examples):
- AFF (Affective): for unwilled experiences like feeling cold, sneezing, emotions
- ERG (Ergative): for deliberate agents performing actions
- ABS (Absolutive): for entities undergoing change
- INS (Instrumental): for tools/instruments being used
- THM (Thematic): for neutral content/topics

VALID FUNCTIONS (only these two):
- STA (Static): for states, conditions, non-changing situations
- DYN (Dynamic): for actions, changes, motion

KEY RULES:
- AFF case REQUIRES STA function (experiences are states, not actions)
- ERG case REQUIRES DYN function (agents perform actions)
- INS case REQUIRES DYN function (instruments are actively used)

RELEVANT GRAMMAR FROM OFFICIAL DOCUMENTATION:
{rag_context}

English description: "{user_input}"

Respond with ONLY valid JSON:
{{"case": "THREE_LETTER_CODE", "function": "STA_or_DYN", "reasoning": "why"}}
"""

    def _build_retry_prompt(self, user_input: str, previous_attempt: Dict, error_msg: str, rag_context: str) -> str:
        """Build prompt for retry after validation failure"""
        return f"""Your previous suggestion was INVALID. Try again.

VALID FUNCTIONS ARE ONLY: STA or DYN (not EXPERIENCER, not semantic roles)
VALID CASES ARE: AFF, ERG, ABS, INS, THM, etc. (three-letter codes)

GRAMMAR RULES:
{rag_context}

REJECTION REASON: {error_msg}

English description: "{user_input}"

Respond with ONLY valid JSON:
{{"case": "THREE_LETTER_CODE", "function": "STA_or_DYN", "reasoning": "why"}}
"""
    
    def _retrieve_context(self, user_input: str) -> str:
        """Use RAG to get relevant grammar rules"""
        # Query RAG with user input
        chunks = self.rag.retrieve(user_input, n_results=3)
        
        if not chunks:
            return "No specific rules found."
        
        # Format as context
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"- {chunk.case_code} ({chunk.case_name}): {chunk.semantic_role}\n"
                f"  {chunk.description[:200]}"
            )
        
        return "\n\n".join(context_parts)

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        # Try to find JSON in response
        try:
            # Look for JSON pattern
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        return None

    def suggest_without_rag(self, user_input: str) -> Tuple[Dict, bool, str]:
        """
        Get LLM suggestion WITHOUT RAG (baseline for comparison)
        
        Returns: (suggestion_dict, is_valid, error_or_success_msg)
        """
        self.llm.reset_conversation()
        
        prompt = self._build_prompt_without_rag(user_input)
        response = self.llm.chat(prompt, max_tokens=200, temperature=0.3)
        
        suggestion = self._parse_llm_response(response)
        if not suggestion:
            return {"raw": response}, False, "Failed to parse LLM response"
        
        # Validate
        result = self.validator.validate(suggestion)
        
        if result.passed:
            return suggestion, True, f"Valid (confidence: {result.confidence:.2f})"
        else:
            error_msg = result.errors[0].message if result.errors else "Unknown error"
            return suggestion, False, error_msg

    def suggest_with_rag(self, user_input: str, max_retries: int = 1) -> Tuple[Dict, bool, str, int]:
        """
        Get LLM suggestion WITH RAG context
        
        Returns: (suggestion_dict, is_valid, message, attempts_used)
        """
        self.llm.reset_conversation()
        
        # Get RAG context
        rag_context = self._retrieve_context(user_input)
        
        # First attempt
        prompt = self._build_prompt_with_rag(user_input, rag_context)
        response = self.llm.chat(prompt, max_tokens=200, temperature=0.3)
        
        suggestion = self._parse_llm_response(response)
        if not suggestion:
            return {"raw": response}, False, "Failed to parse LLM response", 1
        
        # Validate
        result = self.validator.validate(suggestion)
        
        if result.passed:
            return suggestion, True, f"Valid (confidence: {result.confidence:.2f})", 1
        
        # Retry with feedback if allowed
        if max_retries > 0:
            error_msg = result.errors[0].message if result.errors else "Unknown error"
            
            retry_prompt = self._build_retry_prompt(
                user_input, suggestion, error_msg, rag_context
            )
            retry_response = self.llm.chat(retry_prompt, max_tokens=200, temperature=0.2)
            
            retry_suggestion = self._parse_llm_response(retry_response)
            if retry_suggestion:
                retry_result = self.validator.validate(retry_suggestion)
                if retry_result.passed:
                    return retry_suggestion, True, f"Valid after retry (confidence: {retry_result.confidence:.2f})", 2
                else:
                    error_msg = retry_result.errors[0].message if retry_result.errors else "Unknown"
                    return retry_suggestion, False, error_msg, 2
        
        error_msg = result.errors[0].message if result.errors else "Unknown error"
        return suggestion, False, error_msg, 1


# Quick test
if __name__ == "__main__":
    grammar_file = Path("../data/grammar_chunks.json")
    if not grammar_file.exists():
        grammar_file = Path("data/grammar_chunks.json")
    
    pipeline = CopilotPipeline(grammar_file)
    
    test_input = "experiencing cold involuntarily"
    
    print("\n" + "=" * 60)
    print(f"Input: '{test_input}'")
    print("=" * 60)
    
    print("\n--- WITHOUT RAG ---")
    suggestion, valid, msg = pipeline.suggest_without_rag(test_input)
    print(f"Suggestion: {suggestion}")
    print(f"Valid: {valid}")
    print(f"Message: {msg}")
    
    print("\n--- WITH RAG ---")
    suggestion, valid, msg, attempts = pipeline.suggest_with_rag(test_input)
    print(f"Suggestion: {suggestion}")
    print(f"Valid: {valid}")
    print(f"Message: {msg}")
    print(f"Attempts: {attempts}")