import torch

def check_gpu():
    print("\nPyTorch version:", torch.__version__)
    print("\nCUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("\nCUDA version:", torch.version.cuda)
        print("\nCUDA device count:", torch.cuda.device_count())
        print("\nCurrent CUDA device:", torch.cuda.current_device())
        print("\nDevice name:", torch.cuda.get_device_name(0))
        
        # Test CUDA memory
        print("\nTesting CUDA memory allocation...")
        try:
            x = torch.rand(1000, 1000).cuda()
            print("Successfully allocated test tensor on GPU")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print("Error allocating test tensor:", e)
    else:
        print("\nNo CUDA available. To use GPU, please install PyTorch with CUDA support.")
        print("You can install it using:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu() 