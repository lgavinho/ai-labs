import fitz  # PyMuPDF
from typing import Dict, List
from dataclasses import dataclass
import os
import re
from streamlit.logger import get_logger
from utils import split_paragraphs


logger = get_logger(__name__)

@dataclass
class PDFPage:
    """
    Represents a single page from a PDF document
    """
    page_number: int
    content: str

class RAGService:
    
    def __init__(self):
        """
        Initialize the RAG Service for PDF processing
        """
        pass        


    def get_text_from_pdf_file(self, pdf_path: str) -> str:
        """
        Extracts raw text content from a PDF file

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Raw text content from the PDF
        """
        pdf_data = self.process_pdf(pdf_path)
        raw_text = self.get_full_text_from_pdf_pages(pdf_data)
        # TODO better with the split can be in sections contents like chapters, sections, etc.
        splited = split_paragraphs(raw_text) # It is important to split paragraphs
        logger.info("Raw text size: %d", len(splited))
        logger.info("Raw text: %s", splited)
        return splited

    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Processes a PDF file and extracts raw text content
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Dictionary containing the raw text content and metadata
                 Format: {
                     'pages': List[PDFPage],
                     'total_pages': int,
                     'file_path': str
                 }
        """
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            pages = []
            for page_num, page in enumerate(doc):
                content = page.get_text()
                pages.append(PDFPage(
                    page_number=page_num + 1,
                    content=content
                ))
            
            return {
                "pages": pages,
                "total_pages": len(doc),
                "file_path": pdf_path
            }
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        finally:
            if 'doc' in locals():
                doc.close()

    def get_full_text_from_pdf_pages(self, pdf_result: Dict) -> str:
        """
        Concatenates all pages content into a single text string
        
        Args:
            pdf_result (Dict): The result dictionary from process_pdf method
            
        Returns:
            str: Complete text content from all pages
        """
        return "\n\n".join([page.content for page in pdf_result["pages"]])


# Example usage:
# rag_service = RAGService()
# current_dir = os.path.dirname(os.path.abspath(__file__))
# pdf_path = f"{current_dir}/../fixtures/example1.pdf"
# # Get clean text
# clean_text = rag_service.get_text_from_pdf_file(pdf_path)
# print(f"Raw text size: {len(clean_text)}")

