import os
import torch
import torchaudio
import argparse
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import time

from .preprocessing import get_audio_files
from .inference import process_audio
from .config import SAMPLE_RATE

def batch_process(
    input_dir: str,
    output_dir: str,
    model_path: str,
    use_gpu: bool = True,
    num_workers: int = 1,
    debug: bool = False
):
    """
    Process all audio files in a directory using the trained model
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save processed files
        model_path: Path to the trained model
        use_gpu: Whether to use GPU for processing
        num_workers: Number of parallel processes (1 for sequential)
        debug: Whether to generate additional debug files
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files from input directory
    audio_files = get_audio_files(input_dir)
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Print additional info about the files
    if len(audio_files) > 0:
        example_file = audio_files[0]
        try:
            audio, sr = torchaudio.load(example_file)
            duration = audio.shape[1] / sr
            print(f"Example file duration: {duration:.2f} seconds")
        except Exception:
            pass
    
    # Track successful and failed files
    successful_files = []
    failed_files = []
    
    # Force sequential processing if issues with device synchronization
    if use_gpu and "CUDA_LAUNCH_BLOCKING" not in os.environ:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Define the worker function for parallel processing
    def process_file(file_path):
        try:
            # Create output filename
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"isolated_{filename}")
            
            # Force synchronize CUDA for stability in multi-threaded environment
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # For better error reporting
            print(f"Starting processing: {filename}")
            
            # Process with less console output
            process_audio(file_path, output_path, model_path, device, verbose=False, debug=debug)
            
            # Verify file was created
            if os.path.exists(output_path):
                # Validate the output file has sound (not all zeros)
                try:
                    audio, _ = torchaudio.load(output_path)
                    max_amplitude = audio.abs().max().item()
                    if max_amplitude < 0.01:
                        print(f"WARNING: Output file {filename} has very low amplitude: {max_amplitude:.6f}")
                        # Still return success but with a warning
                        return (file_path, output_path, True, f"Low amplitude: {max_amplitude:.6f}")
                    
                    print(f"Successfully processed: {filename} (Max amplitude: {max_amplitude:.4f})")
                    return (file_path, output_path, True, None)
                except Exception as e:
                    return (file_path, output_path, False, f"Error validating output: {str(e)}")
            else:
                return (file_path, output_path, False, "Output file not created")
        except Exception as e:
            return (file_path, None, False, str(e))
        finally:
            # Clean up GPU memory after each file
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    start_time = time.time()
    
    # Process files based on number of workers
    if num_workers > 1 and len(audio_files) > 1:
        print(f"Processing {len(audio_files)} files with {num_workers} parallel workers")
        
        # Use ThreadPoolExecutor for parallel processing
        # (ProcessPoolExecutor doesn't work well with CUDA)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file, file_path) for file_path in audio_files]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
                try:
                    result = future.result()
                    if result[2]:  # Success
                        successful_files.append(result)
                    else:
                        failed_files.append(result)
                        print(f"Failed to process {result[0]}: {result[3]}")
                except Exception as e:
                    print(f"Error in processing: {e}")
    else:
        # Process sequentially
        print(f"Processing {len(audio_files)} files sequentially")
        
        for file_path in tqdm(audio_files, desc="Processing files"):
            result = process_file(file_path)
            if result[2]:  # Success
                successful_files.append(result)
            else:
                failed_files.append(result)
                print(f"Failed to process {result[0]}: {result[3]}")
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Successfully processed: {len(successful_files)}/{len(audio_files)} files")
    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
    
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per file: {total_time/len(audio_files):.2f}s")
    
    # Print information about low amplitude files
    low_amplitude_files = [f for f in successful_files if f[3] and 'Low amplitude' in f[3]]
    if low_amplitude_files:
        print(f"\nWARNING: {len(low_amplitude_files)} files have very low amplitude:")
        for f in low_amplitude_files[:5]:  # Show first 5
            print(f"  - {os.path.basename(f[0])}: {f[3]}")
        if len(low_amplitude_files) > 5:
            print(f"  - ...and {len(low_amplitude_files) - 5} more files")
    
    print(f"\nOutput files saved to: {output_dir}")
    
def main():
    parser = argparse.ArgumentParser(description='Batch process audio files with voice isolation')
    parser.add_argument('--input-dir', default='TEST/MIXED', help='Directory containing input audio files')
    parser.add_argument('--output-dir', default='TEST/ISOLATED', help='Directory to save processed files')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU acceleration if available')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (1=sequential)')
    parser.add_argument('--debug', action='store_true', help='Generate additional debug files')
    
    args = parser.parse_args()
    
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        use_gpu=args.gpu,
        num_workers=args.workers,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
