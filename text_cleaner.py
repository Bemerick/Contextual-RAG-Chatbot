import re

def clean_text(text: str) -> str:
    """Enhanced text cleaning for PDF extracted text"""
    if not text:
        return ""
    
    # First, remove any single-character spacing
    text = re.sub(r'(?<=\S)\s(?=\S)', '', text)
    
    # Add spaces after periods if missing
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    
    # Add spaces between camelCase words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Fix spaces around punctuation
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    
    # Fix spaces around hyphens
    text = re.sub(r'\s*-\s*', '-', text)
    
    # Remove special characters while preserving essential punctuation
    text = re.sub(r'[^\w\s.,!?;:()$"\'-]', ' ', text)
    
    # Fix multiple spaces between words
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Fix spaces around parentheses
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace('—', '-')
    
    # Clean up around newlines
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Final cleanup of any remaining multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()
