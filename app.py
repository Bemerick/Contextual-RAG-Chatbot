#Imports

import streamlit as st
import google.generativeai as genai
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
import io
import json
import logging
from text_cleaner import clean_text
from datetime import datetime
import math
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib
import re
import os
import time
import traceback



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_settings(
    temperature: float,
    context_window: int,
    chunk_size: int,
    chunk_overlap: int,
    top_k_results: int
) -> Tuple[bool, Optional[str]]:
    """
    Validate all settings parameters with detailed error messages

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        # Temperature validation
        if not isinstance(temperature, (int, float)):
            return False, "Temperature must be a number"
        if temperature < 0 or temperature > 1:
            return False, "Temperature must be between 0 and 1"

        # Context window validation
        if not isinstance(context_window, int):
            return False, "Context window must be an integer"
        if context_window < 1:
            return False, "Context window must be at least 1"
        if context_window > 20:  # Practical upper limit
            return False, "Context window cannot exceed 20"

        # Chunk size validation
        if not isinstance(chunk_size, int):
            return False, "Chunk size must be an integer"
        if chunk_size < 100:  # Minimum meaningful chunk
            return False, "Chunk size must be at least 100 characters"
        if chunk_size > 5000:  # Practical upper limit
            return False, "Chunk size cannot exceed 5000 characters"

        # Chunk overlap validation
        if not isinstance(chunk_overlap, int):
            return False, "Chunk overlap must be an integer"
        if chunk_overlap < 0:
            return False, "Chunk overlap cannot be negative"
        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than chunk size"

        # Top K chunks validation
        if not isinstance(top_k_results, int):
            return False, "Top K Chunks must be an integer"
        if top_k_results < 1:
            return False, "Top K chunks must be at least 1"
        if top_k_results > 20:  # Practical upper limit
            return False, "Top K chunks cannot exceed 20"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"

def initialize_session_state():
    """
    Initialize session state with enhanced error handling and validation
    """
    try:
        # Initialize all required session state variables if they don't exist
        if 'document_store' not in st.session_state:
            st.session_state['document_store'] = {}

        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        # Initialize processing status tracking
        if 'processing_status' not in st.session_state:
            st.session_state['processing_status'] = {
                'is_processing': False,
                'current_file': None,
                'progress': 0,
                'errors': []
            }

        # Initialize settings with validation
        default_settings = {
            'model_temperature': 0.7,
            'context_window': 3,
            'chunk_size': 500,
            'chunk_overlap': 100,
            'top_k_results': 5
        }

        # Initialize all settings using dict access
        for key, value in default_settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Validate current settings using get() for safe access
        is_valid, error_msg = validate_settings(
            st.session_state.get('model_temperature', 0.7),
            st.session_state.get('context_window', 3),
            st.session_state.get('chunk_size', 500),
            st.session_state.get('chunk_overlap', 100),
            st.session_state.get('top_k_results', 5)
        )

        if not is_valid:
            logger.warning(f"Invalid settings detected: {error_msg}")
            reset_settings()

    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error("Error initializing application. Please refresh the page.")

def reset_settings():
    """Reset settings to default values with cleanup"""
    settings_keys = [
        'model_temperature',
        'context_window',
        'chunk_size',
        'chunk_overlap',
        'top_k_results'
    ]

    for key in settings_keys:
        if key in st.session_state:
            del st.session_state[key]

class SimplePDFProcessor:
    """Simple and reliable PDF processing class that prioritizes reliability over completeness"""

    def __init__(self, page_timeout: int = 5):
        self.page_timeout = page_timeout  # Timeout in seconds per page

    def process_pdf(self, pdf_file: io.BytesIO) -> Dict[str, Any]:
        """
        Process PDF files using a simple, reliable approach that won't get stuck
        """
        try:
            # Reset the file pointer
            pdf_file.seek(0)
            
            # Log the start of processing
            logger.info(f"Starting to process PDF file")
            
            # Create a PDF reader
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {total_pages} pages")
            except Exception as e:
                logger.error(f"Failed to create PDF reader: {str(e)}")
                raise PDFProcessingError(f"Could not read PDF file: {str(e)}")
            
            if total_pages == 0:
                raise PDFProcessingError("Empty PDF file detected")

            # Extract basic metadata
            try:
                metadata = self._extract_basic_metadata(pdf_reader)
                logger.info(f"Extracted metadata: {metadata}")
            except Exception as e:
                logger.error(f"Metadata extraction failed: {str(e)}")
                metadata = {'pages': total_pages, 'title': 'Unknown', 'error': str(e)}
            
            # Process pages sequentially with strict timeout
            text_chunks = []
            max_pages_to_process = min(50, total_pages)  # Limit to first 50 pages for reliability
            
            for page_num in range(max_pages_to_process):
                # Update progress
                if 'processing_status' in st.session_state:
                    progress = (page_num + 1) / max_pages_to_process
                    st.session_state.processing_status['progress'] = progress
                
                # Process with timeout
                logger.info(f"Processing page {page_num + 1} of {max_pages_to_process}")
                page_text = self._extract_page_text_with_timeout(pdf_reader, page_num)
                
                # Add to chunks if not empty
                if page_text and page_text.strip():
                    text_chunks.append(f"[Page {page_num + 1}]\n{page_text}")
                    logger.info(f"Successfully extracted text from page {page_num + 1}")
                else:
                    text_chunks.append(f"[Page {page_num + 1}]\n[No text content extracted]")
                    logger.warning(f"No text content extracted from page {page_num + 1}")
            
            # Add a note if we didn't process all pages
            if total_pages > max_pages_to_process:
                text_chunks.append(f"[Note: Only processed first {max_pages_to_process} of {total_pages} pages for reliability]")
                logger.info(f"Limited processing to {max_pages_to_process} of {total_pages} pages")
            
            return {
                'text_chunks': text_chunks,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}\n{traceback.format_exc()}")
            raise PDFProcessingError(f"Error processing PDF: {str(e)}")

    def _extract_basic_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """Extract basic metadata with error handling"""
        metadata = {
            'pages': len(pdf_reader.pages),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Try to extract standard metadata fields with fallbacks
        try:
            if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                metadata.update({
                    'title': pdf_reader.metadata.get('/Title', 'Unknown'),
                    'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                    'creation_date': str(pdf_reader.metadata.get('/CreationDate', 'Unknown')),
                })
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata: {str(e)}")
            metadata.update({
                'title': 'Unknown',
                'author': 'Unknown',
                'creation_date': 'Unknown',
            })
            
        return metadata

    def _extract_page_text_with_timeout(self, pdf_reader: PyPDF2.PdfReader, page_num: int) -> str:
        """Extract text from a page with a strict timeout"""
        result = ""
        start_time = datetime.now()
        
        try:
            # Get the page
            page = pdf_reader.pages[page_num]
            
            # Simple text extraction with timeout check
            result = page.extract_text()
            
            # Check if we exceeded timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.page_timeout:
                logger.warning(f"Page {page_num + 1} extraction took {elapsed:.1f}s which exceeds timeout of {self.page_timeout}s")
                return f"[Text extraction timeout for page {page_num + 1}]"
                
            # Clean the text (simple version)
            result = self._clean_text_simple(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
            return f"[Error extracting text from page {page_num + 1}: {str(e)}]"
            
    def _clean_text_simple(self, text: str) -> str:
        """Simple text cleaning that won't hang"""
        if not text:
            return ""
            
        # Basic cleaning only - avoid complex operations that might hang
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.warning(f"Text cleaning error: {str(e)}")
            return text  # Return original text if cleaning fails

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better formatting preservation"""
        return clean_text(text)

class EnhancedRAGChatbot:
    def __init__(self):
        # Get API key from environment variable
        api_key = st.secrets['GOOGLE_API_KEY']
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in streamlit secrets. Please check your .streamlit/secrets.toml file.")
            
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Only use the working model based on testing
        self.model_names = [
            'gemini-1.5-pro',  # Working model confirmed by testing
        ]
        
        # Try to initialize with a working model
        self.text_model = None
        for model_name in self.model_names:
            try:
                logger.info(f"Attempting to initialize model: {model_name}")
                model = genai.GenerativeModel(model_name)
                # Test the model with a simple query
                if self._test_model(model):
                    self.text_model = model
                    logger.info(f"Successfully initialized model: {model_name}")
                    break
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name}: {str(e)}")
        
        # If no models worked, use the first one anyway and hope for the best
        if not self.text_model:
            logger.warning("Could not verify any models. Using first model as fallback.")
            self.text_model = genai.GenerativeModel(self.model_names[0])
            
        # Initialize PDF processor
        self.pdf_processor = SimplePDFProcessor()
        
        # Apply validated temperature setting with safe access
        self.generation_config = {
            "temperature": st.session_state.get('model_temperature', 0.7),
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,  # Reasonable limit for responses
        }
        
    def _test_model(self, model):
        """Test if the model is working properly"""
        try:
            # Simple test query
            test_response = model.generate_content("Hello, are you working? Reply with 'yes' if you are.")
            
            # Check if we got a valid response
            if hasattr(test_response, 'text') and test_response.text and 'yes' in test_response.text.lower():
                return True
                
            logger.warning(f"Model test response doesn't contain 'yes': {test_response}")
            return False
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False

    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Simple and reliable PDF processing"""
        try:
            # Process PDF with progress tracking
            st.session_state.processing_status['is_processing'] = True
            st.session_state.processing_status['current_file'] = pdf_file.name

            # Get file size
            pdf_file.seek(0, 2)
            file_size = pdf_file.tell()
            pdf_file.seek(0)

            # Check file size (100MB limit as example)
            if file_size > 100 * 1024 * 1024:
                raise PDFProcessingError("File size exceeds 100MB limit")

            # Process PDF using the simpler, more reliable method
            result = self.pdf_processor.process_pdf(pdf_file)

            # Update metadata with file size
            if 'metadata' in result and isinstance(result['metadata'], dict):
                result['metadata']['file_size'] = file_size

            return result

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
            raise

        finally:
            st.session_state.processing_status['is_processing'] = False
            st.session_state.processing_status['current_file'] = None
            st.session_state.processing_status['progress'] = 0

    def chunk_text(self, text_chunks: List[str]) -> List[str]:
        """Enhanced text chunking with better context preservation"""
        chunk_size = st.session_state.chunk_size
        overlap = st.session_state.chunk_overlap

        final_chunks = []

        for text in text_chunks:
            # Split into sentences while preserving page markers
            sentences = re.split(r'(?<=[.!?])\s+(?!Page|$$Page)', text)

            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence)

                if current_length + sentence_length > chunk_size:
                    # Add current chunk if not empty
                    if current_chunk:
                        final_chunks.append(' '.join(current_chunk))

                    # Handle overlap
                    if overlap > 0:
                        # Calculate how many sentences to keep for overlap
                        overlap_tokens = []
                        overlap_length = 0
                        for s in reversed(current_chunk):
                            if overlap_length + len(s) <= overlap:
                                overlap_tokens.insert(0, s)
                                overlap_length += len(s)
                            else:
                                break
                        current_chunk = overlap_tokens
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0

                current_chunk.append(sentence)
                current_length += sentence_length

            # Add the last chunk if not empty
            if current_chunk:
                final_chunks.append(' '.join(current_chunk))

        return final_chunks

    def generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """Enhanced embedding generation with batch processing, retry logic and timeout handling"""
        embeddings = []
        max_retries = 3
        batch_size = 10  # Process embeddings in batches of 10
        total_chunks = len(chunks)
        
        # Process in batches to avoid timeouts
        for batch_idx in range(0, total_chunks, batch_size):
            # Update progress
            progress = min(0.95, batch_idx / total_chunks)  # Cap at 95% until complete
            if 'processing_status' in st.session_state:
                st.session_state.processing_status['progress'] = progress
                
            end_idx = min(batch_idx + batch_size, total_chunks)
            batch_chunks = chunks[batch_idx:end_idx]
            
            logger.info(f"Processing embedding batch {batch_idx}-{end_idx} of {total_chunks} chunks")
            
            # Process each chunk in the batch
            batch_embeddings = []
            for i, chunk in enumerate(batch_chunks):
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Truncate chunk if too long (model limit is typically 2048 tokens)
                        if len(chunk) > 8192:  # Approximate character limit
                            chunk = chunk[:8192]
                            
                        # Add timeout handling
                        start_time = datetime.now()
                        timeout_seconds = 30  # 30 second timeout per embedding
                        
                        # Generate embedding with proper error handling for different API versions
                        try:
                            result = genai.embed_content(
                                model="models/embedding-001",
                                content=chunk,
                                task_type="retrieval_document"
                            )
                            
                            # Check if we've exceeded timeout
                            elapsed = (datetime.now() - start_time).total_seconds()
                            if elapsed > timeout_seconds:
                                raise TimeoutError(f"Embedding generation timed out after {elapsed} seconds")

                            # Handle different response formats
                            if isinstance(result, dict) and 'embedding' in result:
                                # Dictionary format
                                embedding = np.array(result['embedding'])
                            elif hasattr(result, 'embedding'):
                                # Object format
                                embedding = np.array(result.embedding)
                            else:
                                # Log the actual result for debugging
                                logger.error(f"Unexpected embedding result format: {type(result)}, {result}")
                                raise ValueError(f"Unexpected embedding result format: {type(result)}")
                                
                            # Log embedding shape for debugging
                            logger.info(f"Generated embedding with shape: {embedding.shape}")
                        except Exception as embed_error:
                            logger.error(f"Error in embed_content: {str(embed_error)}")
                            raise

                        # Validate embedding
                        if embedding.size == 0:
                            raise ValueError("Empty embedding generated")

                        batch_embeddings.append(embedding)
                        break

                    except TimeoutError as te:
                        logger.warning(f"Timeout generating embedding for chunk {batch_idx + i}: {str(te)}")
                        # Create a fallback embedding (zeros) for timed out chunks
                        if result and 'embedding' in result and len(result['embedding']) > 0:
                            # Use partial result if available
                            batch_embeddings.append(np.array(result['embedding']))
                        else:
                            # Create a zero embedding as fallback (same dimension as others)
                            dim = 768  # Default embedding dimension
                            if len(batch_embeddings) > 0:
                                dim = batch_embeddings[0].shape[0]
                            batch_embeddings.append(np.zeros(dim))
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            logger.error(f"Failed to generate embedding after {max_retries} attempts: {str(e)}")
                            # Create a fallback embedding (zeros) for failed chunks
                            dim = 768  # Default embedding dimension
                            if len(batch_embeddings) > 0:
                                dim = batch_embeddings[0].shape[0]
                            batch_embeddings.append(np.zeros(dim))
                        else:
                            logger.warning(f"Retry {retry_count} for embedding generation: {str(e)}")
            
            # Add batch embeddings to overall embeddings list
            embeddings.extend(batch_embeddings)
            
        return embeddings

    def retrieve_relevant_context(self, query: str) -> List[str]:
        """Enhanced context retrieval with better relevance scoring"""
        try:
            # Process query
            processed_query = self.pdf_processor._clean_text(query)

            # Include conversation history in context
            conversation_history = st.session_state.get('conversation_history', [])
            context_window = st.session_state.get('context_window', 3)
            if conversation_history:
                recent_messages = conversation_history[-context_window:]
                context_text = " ".join([msg['content'] for msg in recent_messages])
                processed_query = f"{context_text} {processed_query}"

            # Generate query embedding with proper error handling for different API versions
            try:
                logger.info(f"Generating embedding for query: {processed_query[:50]}...")
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=processed_query,
                    task_type="retrieval_query"
                )
                
                # Handle different response formats
                if isinstance(result, dict) and 'embedding' in result:
                    # Dictionary format
                    query_embedding = np.array(result['embedding'])
                elif hasattr(result, 'embedding'):
                    # Object format
                    query_embedding = np.array(result.embedding)
                else:
                    # Log the actual result for debugging
                    logger.error(f"Unexpected query embedding result format: {type(result)}, {result}")
                    raise ValueError(f"Unexpected query embedding result format: {type(result)}")
                    
                logger.info(f"Generated query embedding with shape: {query_embedding.shape}")
            except Exception as e:
                logger.error(f"Error generating query embedding: {str(e)}\n{traceback.format_exc()}")
                raise

            # Gather all document embeddings and chunks
            all_embeddings = []
            all_chunks = []
            chunk_sources = []

            document_store = st.session_state.get('document_store', {})
            for doc_id, doc_data in document_store.items():
                all_embeddings.extend(doc_data['embeddings'])
                all_chunks.extend(doc_data['chunks'])
                chunk_sources.extend([doc_data['filename']] * len(doc_data['chunks']))

            if not all_embeddings:
                return []

            # Calculate similarities with improved scoring
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                np.array(all_embeddings)
            )[0]

            # Enhanced chunk selection with diversity consideration
            top_k = min(st.session_state.top_k_results * 2, len(similarities))  # Get more candidates for diversity
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Filter out low-relevance chunks and ensure diversity
            min_similarity_threshold = 0.2  # Lower threshold to get more diverse content
            relevant_chunks = []
            seen_sources = set()  # Track sources to ensure diversity

            # First pass: collect chunks by source
            source_chunks = {}
            for idx in top_indices:
                if similarities[idx] >= min_similarity_threshold:
                    chunk_text = all_chunks[idx]
                    source = chunk_sources[idx]
                    
                    if source not in source_chunks:
                        source_chunks[source] = []
                    
                    # Store the chunk with its similarity score
                    source_chunks[source].append((similarities[idx], chunk_text))
            
            # Second pass: select diverse chunks from different sources
            final_top_k = st.session_state.top_k_results
            sources_list = list(source_chunks.keys())
            
            # If we have fewer sources than requested chunks, we'll take multiple from each source
            chunks_per_source = max(1, final_top_k // max(1, len(sources_list)))
            
            # Ensure we get chunks from different sources when possible
            for source in sources_list:
                # Sort chunks for this source by similarity score
                source_chunks[source].sort(reverse=True)
                
                # Take the top N chunks from this source
                for i, (score, chunk_text) in enumerate(source_chunks[source]):
                    if i < chunks_per_source and len(relevant_chunks) < final_top_k:
                        relevant_chunks.append(f"[Source: {source}]\n{chunk_text}")
                        
            # If we still need more chunks, take the highest scoring remaining chunks
            if len(relevant_chunks) < final_top_k:
                # Flatten all remaining chunks
                remaining_chunks = []
                for source in sources_list:
                    for i, (score, chunk_text) in enumerate(source_chunks[source]):
                        if i >= chunks_per_source:  # Only consider chunks we haven't already taken
                            remaining_chunks.append((score, chunk_text, source))
                
                # Sort by score
                remaining_chunks.sort(reverse=True)
                
                # Add highest scoring chunks until we reach the desired count
                for score, chunk_text, source in remaining_chunks:
                    if len(relevant_chunks) < final_top_k:
                        relevant_chunks.append(f"[Source: {source}]\n{chunk_text}")
                    else:
                        break

            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def generate_response(self, query: str, context: List[str]) -> str:
        """Enhanced response generation with better context utilization and error handling"""
        try:
            # Log the start of response generation for debugging
            logger.info(f"Starting to generate response for query: {query[:50]}...")
            logger.info(f"Found {len(context)} context chunks for the query")
            
            # First try the simple keyword-based fallback method if we have context
            if context:
                simple_response = self._generate_simple_response(query, context)
                if simple_response:
                    logger.info("Generated response using simple keyword-based method")
                    return simple_response
            
            # Prepare conversation context
            recent_messages = []
            conversation_history = st.session_state.get('conversation_history', [])
            context_window = st.session_state.get('context_window', 3)
            if conversation_history:
                history_window = conversation_history[-context_window:]
                recent_messages = [
                    f"{msg['role']}: {msg['content']}"
                    for msg in history_window
                ]

            conversation_context = "\n".join(recent_messages) if recent_messages else "No previous context"

            # Prepare context text with better formatting - limit context size to avoid token limits
            max_context_chunks = 3  # Limit to top 3 chunks to avoid exceeding token limits
            limited_context = context[:max_context_chunks] if len(context) > max_context_chunks else context
            
            if len(context) > max_context_chunks:
                logger.info(f"Limiting context from {len(context)} to {max_context_chunks} chunks to avoid token limits")
                
            context_text = "\n\n".join(limited_context) if limited_context else "No relevant context found"
            
            # Log context size for debugging
            logger.info(f"Context text size: {len(context_text)} characters")

            # Enhanced prompt template with better instructions
            prompt = f"""
            You are a helpful AI assistant that provides accurate and comprehensive answers based on the provided context.
            
            CONTEXT INFORMATION:
            {context_text}
            
            USER QUESTION: {query}
            
            INSTRUCTIONS:
            1. Answer the question based ONLY on the context provided above
            2. If the context doesn't contain relevant information, say you don't have enough information
            3. Provide a detailed and informative response that directly addresses the question
            4. Do not repeat the same information multiple times
            5. Do not include phrases like 'Based on the document' or 'According to the context' in your response
            
            ANSWER:
            """.strip()

            # Log prompt size for debugging
            logger.info(f"Prompt size: {len(prompt)} characters")

            # Generate response with error handling and retry logic
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    logger.info(f"Attempt {retry_count + 1} to generate response")
                    
                    # Use a simpler generation config with lower temperature for more reliable responses
                    simple_config = {
                        "temperature": 0.1,  # Very low temperature for more deterministic responses
                        "max_output_tokens": 512,  # Shorter responses to avoid issues
                        "top_p": 0.95,
                        "top_k": 40,
                    }
                    
                    # Generate the response
                    response = self.text_model.generate_content(
                        prompt,
                        generation_config=simple_config
                    )
                    
                    # Debug the response object
                    logger.info(f"Response type: {type(response)}")
                    
                    # Handle different response formats based on API version
                    if hasattr(response, 'text'):
                        # New API format
                        if response.text and response.text.strip():
                            logger.info("Successfully generated response using response.text")
                            return response.text.strip()
                        else:
                            logger.warning("Empty response.text received")
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Alternative format
                        content = response.candidates[0].content
                        if content and hasattr(content, 'parts') and content.parts:
                            text = content.parts[0].text
                            if text and text.strip():
                                logger.info("Successfully generated response using candidates format")
                                return text.strip()
                    elif hasattr(response, 'parts') and response.parts:
                        # Another possible format
                        text = response.parts[0].text
                        if text and text.strip():
                            logger.info("Successfully generated response using parts format")
                            return text.strip()
                    
                    # If we got here, we couldn't extract text from the response
                    logger.error(f"Could not extract text from response: {response}")
                    raise ValueError("Could not extract text from response")

                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Retry {retry_count} for response generation: {str(e)}")
                    if retry_count == max_retries:
                        # If all retries fail, try the simple method again as a last resort
                        if context:
                            simple_response = self._generate_simple_response(query, context, force=True)
                            if simple_response:
                                return simple_response
                        logger.error(f"Failed to generate response after {max_retries} attempts: {str(e)}")
                        return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question or uploading a different document."
                    # Wait briefly before retrying
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but an error occurred while generating the response. Please try again with a simpler question."
            
    def _generate_simple_response(self, query: str, context: List[str], force: bool = False) -> str:
        """Generate a simple response based on keyword matching as a fallback method"""
        try:
            # Only use this method if forced or if the query is simple
            if not force and len(query.split()) > 10:
                return None
                
            logger.info("Attempting simple keyword-based response generation")
            
            # Extract keywords from the query
            query_words = set(query.lower().split())
            important_words = [w for w in query_words if len(w) > 3 and w not in {
                'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
                'does', 'do', 'did', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
                'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might',
                'must', 'about', 'above', 'across', 'after', 'against', 'along', 'among',
                'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside',
                'between', 'beyond', 'but', 'by', 'despite', 'down', 'during', 'except',
                'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off',
                'on', 'onto', 'out', 'outside', 'over', 'past', 'since', 'through',
                'throughout', 'till', 'to', 'toward', 'under', 'underneath', 'until',
                'up', 'upon', 'with', 'within', 'without', 'and', 'or', 'the', 'a', 'an'
            }]
            
            if not important_words:
                return None
                
            logger.info(f"Important query words: {important_words}")
            
            # Find the most relevant context chunk based on keyword matching
            best_chunk = None
            best_score = 0
            
            for chunk in context:
                chunk_lower = chunk.lower()
                score = sum(1 for word in important_words if word in chunk_lower)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            
            # If we found a good match, extract relevant sentences
            if best_chunk and best_score >= 1:
                # Extract the source if available
                source = ""
                if "[Source:" in best_chunk:
                    source_part = best_chunk.split("]")[0] + "]"
                    source = f"{source_part}\n"
                    best_chunk = best_chunk.replace(source_part, "").strip()
                
                # Split into sentences and find the most relevant ones
                sentences = re.split(r'(?<=[.!?])\s+', best_chunk)
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(word in sentence.lower() for word in important_words):
                        relevant_sentences.append(sentence)
                
                # Construct a simple response
                if relevant_sentences:
                    response_text = " ".join(relevant_sentences)
                    return f"{source}Based on the document, {response_text}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error in simple response generation: {str(e)}")
            return None
            
    def direct_fallback_response(self, query: str, chunks_with_sources: List[Tuple[str, str]]) -> str:
        """Generate a direct response using keyword matching without relying on the Google API"""
        try:
            logger.info(f"Generating direct fallback response for query: {query}")
            
            # Extract keywords from the query (more permissive than _generate_simple_response)
            query_words = query.lower().split()
            important_words = [w for w in query_words if len(w) > 2 and w not in {
                'the', 'and', 'or', 'but', 'if', 'a', 'an', 'as', 'at', 'by', 'for', 'in',
                'of', 'on', 'to', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would',
                'shall', 'should', 'may', 'might', 'must', 'that', 'this', 'these', 'those'
            }]
            
            if not important_words:
                # If no important words found, use all words longer than 2 characters
                important_words = [w for w in query_words if len(w) > 2]
                
            if not important_words:
                return "I couldn't find any relevant information in the documents. Please try asking a more specific question."
                
            logger.info(f"Important query words: {important_words}")
            
            # Score all chunks
            chunk_scores = []
            for chunk, source in chunks_with_sources:
                chunk_lower = chunk.lower()
                # Calculate score based on word frequency and position
                score = sum(chunk_lower.count(word) for word in important_words)
                # Bonus for having multiple query words
                unique_matches = sum(1 for word in important_words if word in chunk_lower)
                score += unique_matches * 2  # Bonus for diversity of matches
                
                # Store the score along with the chunk and source
                chunk_scores.append((score, chunk, source))
            
            # Sort by score in descending order
            chunk_scores.sort(reverse=True)
            
            # Take top 3 chunks
            top_chunks = chunk_scores[:3] if len(chunk_scores) > 3 else chunk_scores
            
            if not top_chunks or top_chunks[0][0] == 0:
                return "I couldn't find any relevant information in the documents. Please try asking a more specific question."
            
            # Extract relevant sentences from top chunks
            all_relevant_sentences = []
            sources_used = set()
            
            for score, chunk, source in top_chunks:
                if score > 0:
                    # Split into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    
                    # Find relevant sentences
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        if any(word in sentence_lower for word in important_words):
                            # Add the sentence with its source
                            all_relevant_sentences.append((sentence, source))
                            sources_used.add(source)
            
            if not all_relevant_sentences:
                return "I found some potentially relevant documents, but couldn't extract specific information about your question. Please try rephrasing your question."
            
            # Construct the response
            response_parts = []
            
            # Add a more natural introduction that varies based on the query
            if len(sources_used) == 1:
                source_name = list(sources_used)[0]
                intro_options = [
                    f"Here's what I found in '{source_name}':",
                    f"The document '{source_name}' provides this information:",
                    f"From my analysis of '{source_name}':",
                    f"According to '{source_name}':"
                ]
                response_parts.append(random.choice(intro_options))
            else:
                intro_options = [
                    f"After reviewing {len(sources_used)} relevant documents, here's what I found:",
                    f"Based on information from {len(sources_used)} documents:",
                    f"Here's the relevant information from multiple sources:",
                    f"I found information across {len(sources_used)} documents that addresses your question:"
                ]
                response_parts.append(random.choice(intro_options))
            
            # Add relevant sentences grouped by source
            current_source = None
            source_sentences = []
            
            for sentence, source in all_relevant_sentences:
                if source != current_source:
                    # Add previous source sentences if any
                    if source_sentences:
                        response_parts.append(" ".join(source_sentences))
                        source_sentences = []
                    
                    # Add new source header
                    current_source = source
                    response_parts.append(f"\n\nFrom '{source}':")
                
                # Add sentence to current source
                source_sentences.append(sentence)
            
            # Add any remaining sentences
            if source_sentences:
                response_parts.append(" ".join(source_sentences))
            
            # Combine all parts
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in direct fallback response: {str(e)}\n{traceback.format_exc()}")
            return "I encountered an error while searching through the documents. Please try asking a simpler question."

def main():
    """Enhanced main function with better error handling and user experience"""
    try:
        initialize_session_state()

        # Initialize fallback mode flag if it doesn't exist
        if 'use_fallback_mode' not in st.session_state:
            st.session_state['use_fallback_mode'] = False

        st.set_page_config(
            page_title="Enhanced Multi-PDF RAG Chatbot",
            page_icon="📚",
            layout="wide"
        )

        # Main title and description
        col1, col2 = st.columns([1, 3])
        with col2:
            st.title("📚 Document Analysis Assistant")
            st.markdown("""
            An intelligent chatbot for analyzing multiple PDF documents with enhanced processing capabilities
            and advanced context awareness.
            """)

        # Sidebar Configuration
        with st.sidebar:
            # Display logo
            st.image("assets/Longevity-Logo-Horizontal-Color.png", width=250)
            
            st.header("⚙️ Configuration")
            
            # Add fallback mode toggle
            fallback_mode = st.checkbox(
                "Use Simple Fallback Mode", 
                value=st.session_state.get('use_fallback_mode', False),
                help="Enable this if the AI assistant is not responding. This will use a simpler, keyword-based approach instead of the AI model."
            )
            st.session_state.use_fallback_mode = fallback_mode
            
            if fallback_mode:
                st.info("⚠️ Fallback mode is enabled. The assistant will use a simpler, keyword-based approach to answer questions.")
            
            # Context window setting
            context_value = st.slider(
                "Context Window",
                min_value=1,
                max_value=10,
                value=st.session_state.get('context_window', 3),
                step=1,
                help="Number of previous messages to consider"
            )
            st.session_state.context_window = context_value

            # Chunk size setting
            chunk_value = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=st.session_state.get('chunk_size', 500),
                step=50,
                help="Size of text segments for processing"
            )
            st.session_state.chunk_size = chunk_value

            # Chunk overlap setting
            overlap_value = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=min(500, chunk_value - 50),
                value=min(st.session_state.get('chunk_overlap', 100), chunk_value - 50),
                step=10,
                help="Overlap between consecutive chunks"
            )
            st.session_state.chunk_overlap = overlap_value

            # Temperature setting
            temp_value = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('model_temperature', 0.7),
                step=0.1,
                help="Temperature for response generation"
            )
            st.session_state.model_temperature = temp_value

            # Top-K results setting
            top_k_value = st.slider(
                "Top-K Results",
                min_value=1,
                max_value=10,
                value=st.session_state.get('top_k_results', 3),
                step=1,
                help="Number of top results to retrieve"
            )
            st.session_state.top_k_results = top_k_value

            # Validate all settings together
            is_valid, error_message = validate_settings(
                temp_value,
                context_value,
                chunk_value,
                overlap_value,
                top_k_value
            )

            if not is_valid:
                st.sidebar.error(error_message)

            # Reset button with confirmation
            if st.sidebar.button("Reset to Defaults", key="reset_button"):
                reset_settings()
                st.sidebar.success("Settings reset to defaults!")

            # PDF Upload with enhanced handling
            st.header("📄 Document Upload")
            uploaded_files = st.file_uploader(
                "Upload PDF Documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Select one or more PDF files (max 100MB each)"
            )

            # Process Documents Button
            process_button = st.button(
                "Process Documents",
                disabled=not uploaded_files
            )

            # Document Processing
            if process_button and uploaded_files:
                # Create a container for real-time status updates
                status_container = st.empty()
                progress_container = st.empty()
                detail_container = st.empty()
                
                try:
                    chatbot = EnhancedRAGChatbot()
                    processed_count = 0
                    errors = []
                    
                    # Set up progress tracking
                    progress_bar = progress_container.progress(0)
                    total_files = len(uploaded_files)
                    
                    # Initialize processing timer
                    start_time = datetime.now()
                    max_processing_time = 300  # 5 minutes max processing time
                    
                    for i, pdf_file in enumerate(uploaded_files):
                        try:
                            # Check if we've exceeded the total processing time
                            elapsed_time = (datetime.now() - start_time).total_seconds()
                            if elapsed_time > max_processing_time:
                                status_container.warning(f"Processing timeout reached after {elapsed_time:.1f} seconds. Some documents may not be fully processed.")
                                break
                                
                            # Update status display
                            file_name = pdf_file.name
                            status_container.info(f"Processing file {i+1} of {total_files}: {file_name}")
                            detail_container.text(f"Extracting text from PDF...")
                            
                            # Generate document ID
                            doc_id = hashlib.md5(pdf_file.getvalue()).hexdigest()
                            
                            # Set a timeout for each file processing
                            file_start_time = datetime.now()
                            file_timeout = 120  # 2 minutes per file
                            
                            # Process PDF with timeout monitoring
                            pdf_data = chatbot.process_pdf(pdf_file)
                            
                            # Check if we've exceeded the per-file timeout
                            file_elapsed = (datetime.now() - file_start_time).total_seconds()
                            if file_elapsed > file_timeout:
                                detail_container.warning(f"File processing timeout ({file_elapsed:.1f}s). Moving to next file.")
                                continue
                                
                            detail_container.text(f"Chunking text...")
                            chunks = chatbot.chunk_text(pdf_data['text_chunks'])
                            
                            detail_container.text(f"Generating embeddings for {len(chunks)} chunks...")
                            embeddings = chatbot.generate_embeddings(chunks)
                            
                            # Store processed data
                            st.session_state.document_store[doc_id] = {
                                'filename': pdf_file.name,
                                'chunks': chunks,
                                'embeddings': embeddings,
                                'metadata': pdf_data['metadata']
                            }
                            
                            processed_count += 1
                            
                            # Update progress
                            progress = (i + 1) / total_files
                            progress_bar.progress(progress)
                            
                        except TimeoutError as te:
                            errors.append(f"Timeout processing {pdf_file.name}: {str(te)}")
                            detail_container.warning(f"Timeout processing {pdf_file.name}")
                            
                        except Exception as e:
                            errors.append(f"Error processing {pdf_file.name}: {str(e)}")
                            detail_container.error(f"Error processing {pdf_file.name}: {str(e)}")
                            logger.error(f"Document processing error: {str(e)}\n{traceback.format_exc()}")
                            
                        # Add a small delay between files to prevent API rate limiting
                        time.sleep(1)
                    
                    # Show processing results after processing all files
                    if processed_count > 0:
                        st.success(f"Successfully processed {processed_count} document(s)!")
                    if errors:
                        st.error("\n".join(errors))
                        
                except Exception as e:
                    st.error(f"Error during document processing: {str(e)}")
                    logger.error(f"Document processing error: {str(e)}\n{traceback.format_exc()}")
                finally:
                    # Clean up progress display
                    if 'progress_bar' in locals():
                        progress_bar.empty()

            # Clear Options
            st.header("🧹 Clear Options")
            cols = st.columns(2)

            with cols[0]:
                if st.button("Clear Chat", key="clear_chat"):
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.success("Chat cleared!")
                    st.rerun()

            with cols[1]:
                if st.button("Clear Documents", key="clear_docs"):
                    st.session_state.document_store = {}
                    st.success("Documents cleared!")
                    st.rerun()

        # Chat Interface
        st.header("💬 Chat Interface")

        # Display chat messages
        messages = st.session_state.get('messages', [])
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        document_store = st.session_state.get('document_store', {})
        if prompt := st.chat_input(
            "Ask a question about your documents",
            disabled=not document_store
        ):
            # Add user message
            messages = st.session_state.get('messages', [])
            conversation_history = st.session_state.get('conversation_history', [])
            
            messages.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "user", "content": prompt})
            
            st.session_state['messages'] = messages
            st.session_state['conversation_history'] = conversation_history

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                try:
                    chatbot = EnhancedRAGChatbot()
                    
                    # Check if fallback mode is enabled
                    fallback_mode = st.session_state.get('use_fallback_mode', False)
                    
                    if fallback_mode:
                        # Use direct fallback mode that doesn't rely on the Google API
                        with st.status("🔍 Finding relevant information..."):
                            # Get all document chunks
                            document_store = st.session_state.get('document_store', {})
                            all_chunks = []
                            for doc_id, doc_data in document_store.items():
                                all_chunks.extend([(chunk, doc_data['filename']) for chunk in doc_data['chunks']])
                            
                            st.write(f"Searching through {len(all_chunks)} document chunks")
                        
                        with st.status("💭 Generating direct response..."):
                            response = chatbot.direct_fallback_response(prompt, all_chunks)
                            st.write("Response generated successfully!")
                    else:
                        # Use the normal AI-based approach
                        with st.status("🔍 Retrieving relevant context..."):
                            context = chatbot.retrieve_relevant_context(prompt)
                            st.write(f"Found {len(context)} relevant passages")
                        
                        with st.status("💭 Generating response..."):
                            response = chatbot.generate_response(prompt, context)
                            st.write("Response generated successfully!")
                    
                    st.markdown(response)
                    
                    # Show sources if not in fallback mode
                    if not fallback_mode and 'context' in locals() and context:
                        with st.expander("📚 Sources and References", expanded=False):
                            sources = {}
                            for ctx in context:
                                source = ctx.split(']')[0].replace('[Source: ', '')
                                if source not in sources:
                                    sources[source] = []
                                sources[source].append(ctx)

                            for source, passages in sources.items():
                                st.markdown(f"**{source}**")
                                for i, passage in enumerate(passages, 1):
                                    clean_passage = passage.split('\n', 1)[1] if '\n' in passage else passage
                                    st.text(f"Passage {i}: {clean_passage}")

                    # Update conversation history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    logger.error(f"Response generation error: {str(e)}\n{traceback.format_exc()}")

        # Document Statistics
        document_store = st.session_state.get('document_store', {})
        if document_store:
            with st.expander("📑 Document Statistics", expanded=False):
                st.markdown("### Processed Documents Overview")
                
                # Create statistics table
                stats_data = []
                for doc_id, doc_data in document_store.items():
                    stats_data.append({
                        "Filename": doc_data['filename'],
                        "Chunks": len(doc_data['chunks']),
                        "Total Text": sum(len(chunk) for chunk in doc_data['chunks']),
                        "File Size": f"{doc_data['metadata']['file_size'] / 1024:.1f} KB",
                        "Pages": doc_data['metadata']['pages']
                    })
                
                if stats_data:
                    st.dataframe(
                        stats_data,
                        use_container_width=True,
                        hide_index=True
                    )

                # Show aggregate statistics
                total_docs = len(stats_data)
                total_chunks = sum(stat['Chunks'] for stat in stats_data)
                total_pages = sum(stat['Pages'] for stat in stats_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", total_docs)
                with col2:
                    st.metric("Total Chunks", total_chunks)
                with col3:
                    st.metric("Total Pages", total_pages)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()