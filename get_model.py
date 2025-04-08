import torch
from TRSNet import TRSNet
from ptflops import get_model_complexity_info
import time

def get_model(cfg, model_name, img_name=None):
    model = TRSNet(cfg, model_name, img_name).cuda() if img_name else TRSNet(cfg, model_name).cuda()

    param_count = sum(x.numel() for x in model.parameters())
    print(f"Model based on {model_name} has {param_count / 1e6:.4f}M parameters in total")

    # Compute FLOPs and parameter count using ptflops
    input_size = (3, 384, 384) 
    flops, params = get_model_complexity_info(
        model,
        input_size,
        as_strings=True,      
        print_per_layer_stat=False  
    )
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

    # Measure FPS (timing the forward pass)
    model.eval() 
    dummy_input = torch.randn(1, 3, 384, 384).cuda()  

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        
        start_time = time.time()
        for _ in range(100): 
            _ = model(dummy_input)
        elapsed_time = time.time() - start_time

    fps = 100 / elapsed_time
    print(f"FPS: {fps:.2f} frames per second")

    return model.cuda()
