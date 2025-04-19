"""
Batch process multiple audio files using the trained voice isolation model
"""
import os
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

from .config import OUTPUT_DIR
from .inference import process_audio
from .preprocessing import get_audio_files

def batch_process(
    input_dir: str,
    output_dir: str,
    model_path: str,
    use_gpu: bool = True,
    num_workers: int = 1
):
    """
    Process all audio files in a directory using the trained model
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save processed files
        model_path: Path to the trained model
        use_gpu: Whether to use GPU for processing
        num_workers: Number of parallel processes (1 for sequential)
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Get list of audio files
    audio_files = get_audio_files(input_dir)
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Process files
    start_time = time.time()
    total_files = len(audio_files)
    
    print(f"\n=== BATCH VOICE ISOLATION ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_path}")
    print(f"Files to process: {total_files}")
    
    if num_workers > 1 and total_files > 1:
        print(f"Processing in parallel with {num_workers} workers")
        
        # Define a worker function for parallel processing
        def process_file(file_path):
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"isolated_{file_name}")
            
            # Process with less console output
            try:
                process_audio(file_path, output_path, model_path, device, verbose=False)
                return (file_path, output_path, True)
            except Exception as e:
                return (file_path, None, False, str(e))
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_file, f): f for f in audio_files}
            
            # Process results as they complete
            with tqdm(total=total_files, desc="Processing files") as progress_bar:
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result[2]:  # Success
                            progress_bar.set_postfix(file=os.path.basename(result[1]))
                        else:  # Failure
                            progress_bar.set_postfix(file=f"ERROR: {os.path.basename(file_path)}")
                            print(f"\nError processing {file_path}: {result[3]}")
                    except Exception as e:
                        print(f"\nError processing {file_path}: {e}")
                    progress_bar.update(1)
    else:
        # Process sequentially with full output
        for i, file_path in enumerate(tqdm(audio_files, desc="Processing files")):
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"isolated_{file_name}")
            
            print(f"\n[{i+1}/{total_files}] Processing: {file_name}")
            try:
                process_audio(file_path, output_path, model_path, device)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Processed {total_files} files in {total_time:.2f}s")
    print(f"Average time per file: {total_time/total_files:.2f}s")
    print(f"Output saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Batch process audio files with voice isolation')
    parser.add_argument('--input-dir', default='TEST/MIXED', help='Directory containing input audio files')
    parser.add_argument('--output-dir', default='TEST/ISOLATED', help='Directory to save processed files')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU acceleration if available')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (1 for sequential)')
    
    args = parser.parse_args()
    
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        use_gpu=args.gpu,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()
