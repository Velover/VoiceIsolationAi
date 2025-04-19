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

from .config import OUTPUT_DIR, WINDOW_SIZES
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
    
    # Create output directory - ensure it exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
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
    
    # Track successful and failed files
    successful_files = []
    failed_files = []
    
    # Force sequential processing if issues with device synchronization
    if device.type == 'cuda' and total_files > 1:
        num_workers_original = num_workers
        if num_workers > 1:
            print(f"Note: Using GPU with multiple workers can sometimes cause device synchronization issues.")
            print(f"If you encounter errors, try setting --workers 1")
    
    if num_workers > 1 and total_files > 1:
        print(f"Processing in parallel with {num_workers} workers")
        
        # Define a worker function for parallel processing
        def process_file(file_path):
            try:
                # Create output filename
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, f"isolated_{filename}")
                
                # Force synchronize CUDA for stability in multi-threaded environment
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Process with less console output
                process_audio(file_path, output_path, model_path, device, verbose=False)
                
                # Verify file was created
                if os.path.exists(output_path):
                    return (file_path, output_path, True, None)
                else:
                    return (file_path, output_path, False, "Output file not created")
            except Exception as e:
                return (file_path, None, False, str(e))
            finally:
                # Clean up GPU memory after each file
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
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
                            successful_files.append(result[1])
                            progress_bar.set_postfix(file=os.path.basename(result[1]))
                        else:  # Failure
                            failed_files.append((file_path, result[3]))
                            progress_bar.set_postfix(file=f"ERROR: {os.path.basename(file_path)}")
                            print(f"\nError processing {file_path}: {result[3]}")
                    except Exception as e:
                        failed_files.append((file_path, str(e)))
                        print(f"\nError processing {file_path}: {e}")
                    
                    progress_bar.update(1)
                    
                    # Force GPU memory cleanup periodically
                    if device.type == 'cuda' and len(successful_files) % 5 == 0:
                        torch.cuda.empty_cache()
    else:
        # Process sequentially with full output
        for i, file_path in enumerate(tqdm(audio_files, desc="Processing files")):
            try:
                # Force clean GPU memory before each file
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Create output filename
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, f"isolated_{filename}")
                
                print(f"\n[{i+1}/{total_files}] Processing: {filename}")
                print(f"Saving to: {output_path}")
                
                # Process audio file
                process_audio(file_path, output_path, model_path, device)
                
                # Verify file was created
                if os.path.exists(output_path):
                    successful_files.append(output_path)
                else:
                    failed_files.append((file_path, "Output file not created"))
                    print(f"Warning: Output file {output_path} was not created!")
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"Error processing {file_path}: {e}")
                
            # Force synchronize and clear CUDA cache after each file
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    
    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Processed {total_files} files in {total_time:.2f}s")
    
    if total_files > 0:
        print(f"Average time per file: {total_time/total_files:.2f}s")
    
    print(f"Output saved to: {output_dir}")
    print(f"Successfully processed: {len(successful_files)} files")
    
    if successful_files:
        print("\nSample of successful files:")
        for f in successful_files[:5]:
            print(f"  - {os.path.basename(f)}")
        
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for path, error in failed_files:  # Show all errors
            print(f"  - {os.path.basename(path)}: {error}")
            
    # Return list of success/failed for testing
    return successful_files, failed_files

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
