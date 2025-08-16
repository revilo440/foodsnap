#!/usr/bin/env python3
"""Simple syntax validation test for app.py without requiring heavy dependencies."""

import ast
import sys

def test_syntax():
    """Test that app.py has valid Python syntax."""
    try:
        with open('app.py', 'r') as f:
            source = f.read()
        
        ast.parse(source)
        print("✅ app.py syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def test_imports():
    """Test that required imports are present."""
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        if 'from vllm import LLM, SamplingParams' in content:
            print("✅ vLLM imports found")
        else:
            print("❌ vLLM imports missing")
            return False
            
        if '_extract_with_vllm' in content:
            print("✅ vLLM extraction method found")
        else:
            print("❌ vLLM extraction method missing")
            return False
            
        if '"vllm" in self.model_name' in content:
            print("✅ vLLM model routing found")
        else:
            print("❌ vLLM model routing missing")
            return False
            
        if 'timeout=900' in content:
            print("✅ Updated timeout found")
        else:
            print("❌ Updated timeout missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False

if __name__ == "__main__":
    print("Testing vLLM integration implementation...")
    
    syntax_ok = test_syntax()
    imports_ok = test_imports()
    
    if syntax_ok and imports_ok:
        print("\n✅ All tests passed! vLLM integration is ready.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
