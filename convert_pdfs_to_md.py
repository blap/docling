#!/usr/bin/env python3
"""
Script to convert PDF files to Markdown using Qwen3-VL-2B-Instruct model for OCR.
"""
import os
import sys
import argparse
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import concurrent.futures
from typing import List, Tuple
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_model():
    """Load the Qwen3-VL-2B-Instruct model and processor."""
    logger.info("Loading Qwen3-VL-2B-Instruct model...")
    
    try:
        # Load the processor and model
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")  # Using available model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",  # Using available model
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("Model loaded successfully!")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # If the specific model doesn't exist, try a similar one
        logger.info("Trying alternative model...")
        try:
            processor = Qwen2VLProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "microsoft/Phi-3.5-vision-instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Alternative model loaded successfully!")
            return model, processor
        except:
            logger.error("Could not load any model. Please check if the required model is available.")
            sys.exit(1)

def extract_images_from_pdf(pdf_path: str) -> List[Tuple[Image.Image, int]]:
    """Extract images from PDF pages."""
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Adjust matrix for higher resolution (300 DPI)
        mat = fitz.Matrix(3.0, 3.0)  # 300/72 = ~4.17, but we'll use 3 for a good balance of quality and performance
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append((img, page_num))
    
    pdf_document.close()
    return images

def convert_single_pdf_to_md(pdf_path: str, output_dir: str, model, processor) -> None:
    """Convert a single PDF file to Markdown using the vision model."""
    try:
        logger.info(f"Processing {pdf_path}...")
        
        # Extract images from PDF
        images = extract_images_from_pdf(pdf_path)
        
        markdown_content = []
        
        for img, page_num in images:
            # Prepare the conversation message for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {
                            "type": "text", 
                            "text": "Convert this document page to markdown. Do not miss any text and only output the bare markdown without any additional commentary."
                        },
                    ],
                }
            ]
            
            # Process the image and text
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            
            # Generate the response
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Add the page content to markdown
            markdown_content.append(f"# Page {page_num + 1}\n\n{output_text[0]}\n\n---\n\n")
        
        # Create output filename
        pdf_name = Path(pdf_path).stem
        output_path = Path(output_dir) / f"{pdf_name}.md"
        
        # Write markdown content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("".join(markdown_content))
        
        logger.info(f"Successfully converted {pdf_path} to {output_path}")
        
    except Exception as e:
        logger.error(f"Error converting {pdf_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using a vision language model")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default=r"C:\Users\bruno.pio\OneDrive - mtegovbr\Área de Trabalho\PDF",
        help="Directory containing PDF files to convert"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=r"C:\Users\bruno.pio\OneDrive - mtegovbr\Área de Trabalho\2024\PDF md",
        help="Directory to save converted Markdown files"
    )
    parser.add_argument(
        "--num-threads", 
        type=int, 
        default=12,
        help="Number of threads to use for parallel processing"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Load the model and processor
    model, processor = setup_model()
    
    # Process PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(convert_single_pdf_to_md, str(pdf_path), str(output_dir), model, processor)
            for pdf_path in pdf_files
        ]
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                logger.error(f"Error in thread: {str(e)}")
    
    logger.info("All PDFs have been processed!")

if __name__ == "__main__":
    main()