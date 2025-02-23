#Imports

import streamlit as st
import google.generativeai as genai
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
import re
import hashlib
from datetime import datetime
import io
import traceback
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

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
        # Initialize basic session variables
        if 'document_store' not in st.session_state:
            st.session_state.document_store = {}

        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Initialize processing status tracking
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {
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

        for key, value in default_settings.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Validate current settings
        is_valid, error_msg = validate_settings(
            st.session_state.model_temperature,
            st.session_state.context_window,
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
            st.session_state.top_k_results
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

class EnhancedPDFProcessor:
    """Separate class for handling PDF processing with enhanced capabilities"""

    def __init__(self, max_chunk_size: int = 1024 * 1024):  # 1MB default chunk size
        self.max_chunk_size = max_chunk_size

    def process_large_pdf(self, pdf_file: io.BytesIO) -> Dict[str, Any]:
        """
        Process large PDF files in chunks with progress tracking
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)

            if total_pages == 0:
                raise PDFProcessingError("Empty PDF file detected")

            metadata = self._extract_metadata(pdf_reader)
            text_chunks = []

            # Calculate optimal batch size based on available memory
            batch_size = min(20, total_pages)  # Process max 20 pages at once

            for start_idx in range(0, total_pages, batch_size):
                end_idx = min(start_idx + batch_size, total_pages)
                batch_pages = range(start_idx, end_idx)

                # Process batch of pages
                with ThreadPoolExecutor() as executor:
                    future_to_page = {
                        executor.submit(self._process_page, pdf_reader.pages[i], i): i
                        for i in batch_pages
                    }

                    for future in as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            page_text = future.result()
                            if page_text.strip():  # Only add non-empty pages
                                text_chunks.append(f"[Page {page_num + 1}]\n{page_text}")
                        except Exception as e:
                            logger.warning(f"Error processing page {page_num + 1}: {str(e)}")

                # Update progress
                progress = (end_idx / total_pages) * 100
                st.session_state.processing_status['progress'] = progress

            return {
                'text_chunks': text_chunks,
                'metadata': metadata
            }

        except Exception as e:
            raise PDFProcessingError(f"Error processing PDF: {str(e)}")

    def _extract_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """Extract and validate PDF metadata"""
        try:
            metadata = {
                'pages': len(pdf_reader.pages),
                'title': pdf_reader.metadata.get('/Title', 'Unknown'),
                'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                'creation_date': str(pdf_reader.metadata.get('/CreationDate', 'Unknown')),
                'subject': pdf_reader.metadata.get('/Subject', 'Unknown'),
                'keywords': pdf_reader.metadata.get('/Keywords', 'None'),
                'producer': pdf_reader.metadata.get('/Producer', 'Unknown'),
                'file_size': 'Unknown',  # Will be updated with actual size
                'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Sanitize metadata values
            for key, value in metadata.items():
                if not isinstance(value, (str, int)) or value is None:
                    metadata[key] = 'Unknown'

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {'error': 'Metadata extraction failed'}

    def _process_page(self, page: PyPDF2.PageObject, page_num: int) -> str:
        """Process a single PDF page with enhanced text extraction"""
        try:
            text = page.extract_text()

            # Enhanced text cleaning
            text = self._clean_text(text)

            # Basic text validation
            if not text.strip():
                logger.warning(f"Empty text detected on page {page_num + 1}")
                return ""

            return text

        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {str(e)}")
            return f"[Error processing page {page_num + 1}]"

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better formatting preservation"""
        # Remove multiple spaces while preserving paragraph breaks
        text = re.sub(r'\s{2,}', ' ', text)

        # Normalize line breaks
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove special characters while preserving essential punctuation
        text = re.sub(r'[^\w\s.,!?;:()$$$$"\'%-]', '', text)

        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"').replace('—', '-')

        return text.strip()

class EnhancedRAGChatbot:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.text_model = genai.GenerativeModel('gemini-pro')
        self.pdf_processor = EnhancedPDFProcessor()

        # Apply validated temperature setting
        self.generation_config = {
            "temperature": st.session_state.model_temperature,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,  # Reasonable limit for responses
        }

    def process_pdf(self, pdf_file) -> Dict[str, Any]:
        """Enhanced PDF processing with better error handling"""
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

            # Process PDF
            result = self.pdf_processor.process_large_pdf(pdf_file)

            # Update metadata with file size
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
        """Enhanced embedding generation with retry logic and error handling"""
        embeddings = []
        max_retries = 3

        for chunk in chunks:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Truncate chunk if too long (model limit is typically 2048 tokens)
                    if len(chunk) > 8192:  # Approximate character limit
                        chunk = chunk[:8192]

                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=chunk,
                        task_type="retrieval_document"
                    )

                    embedding = np.array(result['embedding'])

                    # Validate embedding
                    if embedding.size == 0:
                        raise ValueError("Empty embedding generated")

                    embeddings.append(embedding)
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Failed to generate embedding after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Retry {retry_count} for embedding generation: {str(e)}")

        return embeddings

    def retrieve_relevant_context(self, query: str) -> List[str]:
        """Enhanced context retrieval with better relevance scoring"""
        try:
            # Process query
            processed_query = self.pdf_processor._clean_text(query)

            # Include conversation history in context
            if st.session_state.conversation_history:
                recent_messages = st.session_state.conversation_history[-st.session_state.context_window:]
                context_text = " ".join([msg['content'] for msg in recent_messages])
                processed_query = f"{context_text} {processed_query}"

            # Generate query embedding
            query_embedding = np.array(
                genai.embed_content(
                    model="models/embedding-001",
                    content=processed_query,
                    task_type="retrieval_query"
                )['embedding']
            )

            # Gather all document embeddings and chunks
            all_embeddings = []
            all_chunks = []
            chunk_sources = []

            for doc_id, doc_data in st.session_state.document_store.items():
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

            # Enhanced chunk selection
            top_k = min(st.session_state.top_k_results, len(similarities))
            top_indices = similarities.argsort()[-top_k:][::-1]

            # Filter out low-relevance chunks
            min_similarity_threshold = 0.3
            relevant_chunks = []

            for idx in top_indices:
                if similarities[idx] >= min_similarity_threshold:
                    chunk_text = all_chunks[idx]
                    source = chunk_sources[idx]
                    relevant_chunks.append(f"[Source: {source}]\n{chunk_text}")

            return relevant_chunks

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def generate_response(self, query: str, context: List[str]) -> str:
        """Enhanced response generation with better context utilization"""
        try:
            # Prepare conversation context
            recent_messages = []
            if st.session_state.conversation_history:
                history_window = st.session_state.conversation_history[-st.session_state.context_window:]
                recent_messages = [
                    f"{msg['role']}: {msg['content']}"
                    for msg in history_window
                ]

            conversation_context = "\n".join(recent_messages) if recent_messages else "No previous context"

            # Prepare context text with better formatting
            context_text = "\n\n".join(context) if context else "No relevant context found"

            # Enhanced prompt template
            prompt = f"""
            You are an advanced document analysis AI assistant skilled in providing precise, context-aware responses.

            Previous Conversation:
            {conversation_context}

            Document Context:
            {context_text}

            Current Question: {query}

            Instructions:
            1. Analyze the provided context thoroughly
            2. Consider the conversation history for better context understanding
            3. If the context is insufficient, clearly state what information is missing
            4. Provide a well-structured, comprehensive answer
            5. Support your response with specific references to the source documents
            6. Use appropriate formatting for better readability
            7. Highlight any uncertainties or assumptions in your response
            8. Maintain consistency with previous responses in the conversation

            Please provide your response:
            """

            # Generate response with error handling and retry logic
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    response = self.text_model.generate_content(
                        prompt,
                        generation_config=self.generation_config
                    )

                    if not response or not response.text:
                        raise ValueError("Empty response generated")

                    return response.text

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        logger.error(f"Failed to generate response after {max_retries} attempts: {str(e)}")
                        return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
                    logger.warning(f"Retry {retry_count} for response generation: {str(e)}")

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but an error occurred while generating the response. Please try again."

def main():
    """Enhanced main function with better error handling and user experience"""
    try:
        initialize_session_state()

        st.set_page_config(
            page_title="Enhanced Multi-PDF RAG Chatbot",
            page_icon="📚",
            layout="wide"
        )

        # Main title and description
        st.title("📚 Document Analysis Assistant")
        st.markdown("""
        An intelligent chatbot for analyzing multiple PDF documents with enhanced processing capabilities
        and advanced context awareness.
        """)

        # Sidebar Configuration
        with st.sidebar:
            st.header("⚙️ Configuration")

            # API Key Input with validation
            api_key = st.text_input("Enter Gemini API Key", type="password")
            if api_key:
                try:
                    # Validate API key
                    genai.configure(api_key=api_key)
                    genai.GenerativeModel('gemini-pro')
                except Exception as e:
                    st.error("Invalid API key. Please check and try again.")
                    api_key = None

            # Advanced Settings
            with st.expander("🔧 Advanced Settings", expanded=False):
                # Temperature setting
                temp_value = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.model_temperature,
                    step=0.1,
                    help="Controls response creativity (0: focused, 1: creative)",
                    key="model_temperature"
                )

                # Context window setting
                context_value = st.slider(
                    "Context Window",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.context_window,
                    step=1,
                    help="Number of previous messages to consider",
                    key="context_window"
                )

                # Chunk size setting
                chunk_value = st.slider(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=st.session_state.chunk_size,
                    step=50,
                    help="Size of text segments for processing",
                    key="chunk_size"
                )

                # Chunk overlap setting
                overlap_value = st.slider(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=min(500, chunk_value - 50),
                    value=min(st.session_state.chunk_overlap, chunk_value - 50),
                    step=10,
                    help="Overlap between consecutive chunks",
                    key="chunk_overlap"
                )

                # Top K chunks setting
                top_k_value = st.slider(
                    "Top K Chunks",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.top_k_results,
                    step=1,
                    help="Number of relevant passages to retrieve",
                    key="top_k_results"
                )

                # Validate settings
                is_valid, error_msg = validate_settings(
                    temp_value,
                    context_value,
                    chunk_value,
                    overlap_value,
                    top_k_value
                )

                if not is_valid:
                    st.error(error_msg)

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
                disabled=not (uploaded_files and api_key)
            )

            # Document Processing
            if process_button and uploaded_files and api_key:
                with st.spinner("Processing documents..."):
                    try:
                        chatbot = EnhancedRAGChatbot(api_key)
                        processed_count = 0
                        errors = []

                        progress_bar = st.progress(0)

                        for i, pdf_file in enumerate(uploaded_files):
                            try:
                                # Update progress
                                progress = (i + 1) / len(uploaded_files)
                                progress_bar.progress(progress)

                                # Generate document ID
                                doc_id = hashlib.md5(pdf_file.getvalue()).hexdigest()

                                # Process PDF
                                pdf_data = chatbot.process_pdf(pdf_file)
                                chunks = chatbot.chunk_text(pdf_data['text_chunks'])
                                embeddings = chatbot.generate_embeddings(chunks)

                                # Store processed data
                                st.session_state.document_store[doc_id] = {
                                    'filename': pdf_file.name,
                                    'chunks': chunks,
                                    'embeddings': embeddings,
                                    'metadata': pdf_data['metadata']
                                }

                                processed_count += 1

                            except Exception as e:
                                errors.append(f"Error processing {pdf_file.name}: {str(e)}")

                        # Show processing results
                        if processed_count > 0:
                            st.success(f"Successfully processed {processed_count} document(s)!")
                        if errors:
                            st.error("\n".join(errors))

                    except Exception as e:
                        st.error(f"Error during document processing: {str(e)}")
                    finally:
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
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input(
            "Ask a question about your documents",
            disabled=not (api_key and st.session_state.document_store)
        ):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.conversation_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display response
            with st.chat_message("assistant"):
                try:
                    chatbot = EnhancedRAGChatbot(api_key)

                    with st.status("🔍 Retrieving relevant context..."):
                        context = chatbot.retrieve_relevant_context(prompt)
                        st.write(f"Found {len(context)} relevant passages")
                    
                    with st.status("💭 Generating response..."):
                        response = chatbot.generate_response(prompt, context)
                        st.write("Response generated successfully!")
                    
                    st.markdown(response)
                    
                    # Show sources
                    if context:
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
        if st.session_state.document_store:
            with st.expander("📑 Document Statistics", expanded=False):
                st.markdown("### Processed Documents Overview")
                
                # Create statistics table
                stats_data = []
                for doc_id, doc_data in st.session_state.document_store.items():
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