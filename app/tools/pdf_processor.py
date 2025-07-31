import PyPDF2
import io
from typing import Dict, Any, List, Optional
import re
from datetime import datetime

from .article_analyzer import ArticleAnalyzer
from .vector_store import VectorStoreService


class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor with article analyzer and vector store"""
        self.article_analyzer = ArticleAnalyzer()
        self.vector_store = VectorStoreService()
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text content from PDF bytes
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Clean up the text
            text = self._clean_extracted_text(text)
            return text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\d+\n', ' ', text)
        text = re.sub(r'\nPage \d+\n', ' ', text)
        
        # Remove URLs and DOIs (basic patterns)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'doi:\s*[\w\./\-]+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_metadata_from_text(self, text: str, filename: str = "") -> Dict[str, Any]:
        """
        Extract basic metadata from PDF text using simple heuristics
        
        Args:
            text: Extracted text content
            filename: Original filename
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "title": "",
            "authors": [],
            "abstract": "",
            "year": None,
            "journal": "",
            "source": "pdf_upload",
            "filename": filename
        }
        
        # Try to extract title (usually in the first few lines)
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) > 20 and len(line) < 200 and not line.startswith('Abstract'):
                # Simple heuristic: title is often the longest line in the beginning
                if len(line) > len(metadata["title"]):
                    metadata["title"] = line
        
        # Try to extract abstract
        abstract_match = re.search(r'(?:Abstract|ABSTRACT)\s*[:\-]?\s*(.*?)(?:\n\s*\n|Keywords|KEYWORDS|1\.|Introduction|INTRODUCTION)', 
                                 text, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Clean up the abstract
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:  # Only use if it's substantial
                metadata["abstract"] = abstract[:2000]  # Limit length
        
        # Try to extract year
        year_matches = re.findall(r'(?:19|20)\d{2}', text)
        if year_matches:
            # Get the most recent reasonable year
            years = [int(y) for y in year_matches if 1990 <= int(y) <= datetime.now().year]
            if years:
                metadata["year"] = max(years)
        
        # Try to extract authors (very basic heuristics)
        # Look for common author patterns
        author_patterns = [
            r'(?:Author[s]?|By)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)\s*$'
        ]
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text[:1000], re.MULTILINE | re.IGNORECASE)
            if matches:
                authors_text = matches[0]
                authors = [author.strip() for author in authors_text.split(',')]
                authors = [author for author in authors if len(author.split()) <= 4]  # Filter out long strings
                if authors:
                    metadata["authors"] = authors
                    break
        
        # If no abstract found, use first substantial paragraph
        if not metadata["abstract"]:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
            if paragraphs:
                metadata["abstract"] = paragraphs[0][:2000]
        
        # Use filename as title if no title found
        if not metadata["title"] and filename:
            metadata["title"] = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        
        return metadata
    
    async def process_pdf(
        self, 
        pdf_content: bytes, 
        filename: str = "",
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF file: extract text, analyze content, and add to knowledge base
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename
            custom_metadata: Optional custom metadata to override extracted metadata
            
        Returns:
            Dictionary with processing results and analysis
        """
        try:
            # Extract text from PDF
            extracted_text = self.extract_text_from_pdf(pdf_content)
            
            if len(extracted_text) < 100:
                raise Exception("Extracted text is too short. PDF might be empty or text extraction failed.")
            
            # Extract metadata
            metadata = self.extract_metadata_from_text(extracted_text, filename)
            
            # Override with custom metadata if provided
            if custom_metadata:
                metadata.update(custom_metadata)
            
            # Use extracted text as abstract if no abstract found
            if not metadata.get("abstract"):
                # Use first 2000 characters as abstract
                metadata["abstract"] = extracted_text[:2000]
            
            # Analyze the content using the article analyzer
            if metadata.get("abstract"):
                analysis, token_usage = await self.article_analyzer.analyze(metadata["abstract"])
                
                # Combine metadata with analysis
                full_article = {**metadata, **analysis}
                
                # Add the full text for better context
                full_article["full_text"] = extracted_text
                
                # Add to vector store
                vector_stats = self.vector_store.add_articles([full_article])
                
                return {
                    "success": True,
                    "article": full_article,
                    "analysis": analysis,
                    "token_usage": token_usage,
                    "vector_stats": vector_stats,
                    "extracted_text_length": len(extracted_text),
                    "metadata": metadata
                }
            else:
                raise Exception("No substantial content found for analysis")
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "extracted_text_length": len(extracted_text) if 'extracted_text' in locals() else 0
            }
    
    def get_pdf_info(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Get basic information about a PDF file
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Dictionary with PDF information
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            info = {
                "num_pages": len(pdf_reader.pages),
                "size_bytes": len(pdf_content),
                "size_mb": round(len(pdf_content) / (1024 * 1024), 2)
            }
            
            # Try to get metadata if available
            if pdf_reader.metadata:
                info.update({
                    "title": pdf_reader.metadata.get('/Title', ''),
                    "author": pdf_reader.metadata.get('/Author', ''),
                    "subject": pdf_reader.metadata.get('/Subject', ''),
                    "creator": pdf_reader.metadata.get('/Creator', ''),
                    "creation_date": pdf_reader.metadata.get('/CreationDate', '')
                })
            
            return info
            
        except Exception as e:
            return {"error": f"Error reading PDF info: {str(e)}"}
    
    def validate_pdf(self, pdf_content: bytes, max_size_mb: int = 50) -> Dict[str, Any]:
        """
        Validate PDF file before processing
        
        Args:
            pdf_content: PDF file content as bytes
            max_size_mb: Maximum allowed file size in MB
            
        Returns:
            Validation result dictionary
        """
        size_mb = len(pdf_content) / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return {
                "valid": False,
                "error": f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)"
            }
        
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if len(pdf_reader.pages) == 0:
                return {
                    "valid": False,
                    "error": "PDF file appears to be empty (no pages found)"
                }
            
            # Try to extract some text to ensure it's readable
            sample_text = ""
            for i, page in enumerate(pdf_reader.pages[:3]):  # Check first 3 pages
                sample_text += page.extract_text()
                if len(sample_text) > 100:  # Found some text
                    break
            
            if len(sample_text.strip()) < 50:
                return {
                    "valid": False,
                    "error": "PDF appears to contain no readable text (might be image-based)"
                }
            
            return {
                "valid": True,
                "num_pages": len(pdf_reader.pages),
                "size_mb": size_mb,
                "sample_text_length": len(sample_text)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Error validating PDF: {str(e)}"
            }