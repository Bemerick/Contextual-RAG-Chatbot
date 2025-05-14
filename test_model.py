import google.generativeai as genai
import sys
import toml
import os

def test_model(model_name, api_key):
    """Test if a specific model is working with the given API key"""
    print(f"\nTesting model: {model_name}")
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        
        # Test with a simple query
        print("Sending test query...")
        response = model.generate_content("Hello, are you working? Reply with a short message.")
        
        # Check the response
        if hasattr(response, 'text') and response.text:
            print(f"SUCCESS! Model responded with: {response.text[:100]}...")
            return True
        else:
            print(f"ERROR: Model returned an empty response: {response}")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def main():
    # Get API key from secrets.toml
    try:
        secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
        secrets = toml.load(secrets_path)
        api_key = secrets.get('GOOGLE_API_KEY')
        
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not found in .streamlit/secrets.toml")
            return
            
        print(f"Found API key: {api_key[:5]}...{api_key[-5:]}")
    except Exception as e:
        print(f"ERROR loading secrets: {str(e)}")
        return
    
    # List of models to test
    models = [
        'gemini-2.5-pro-preview-05-06',  # Current model used in the application
        'gemini-1.5-pro',
        'gemini-1.0-pro',
        'gemini-pro'
    ]
    
    # Test each model
    working_models = []
    for model_name in models:
        if test_model(model_name, api_key):
            working_models.append(model_name)
    
    # Summary
    print("\n=== SUMMARY ===")
    if working_models:
        print(f"Working models: {', '.join(working_models)}")
    else:
        print("No models are working with your API key.")
        print("Possible issues:")
        print("1. The API key may be invalid or expired")
        print("2. You may not have access to these models")
        print("3. The Google Generative AI service may be experiencing issues")
        print("4. There may be network connectivity issues")
        print("\nRecommendation: Use the fallback mode in your application until this is resolved.")

if __name__ == "__main__":
    main()
