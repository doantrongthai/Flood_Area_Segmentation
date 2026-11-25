import copy
import torch
import numpy as np
import time

def calculate_miou(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        all_preds = (all_preds > threshold).astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)
        
        all_preds_flat = all_preds.reshape(-1)
        all_labels_flat = all_labels.reshape(-1)
        
        intersection = np.logical_and(all_preds_flat, all_labels_flat).sum()
        union = np.logical_or(all_preds_flat, all_labels_flat).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)
    
    else:
        preds_class = np.argmax(all_preds, axis=1)
        
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        preds_class_flat = preds_class.reshape(-1)
        labels_flat = all_labels.reshape(-1)
        
        ious = []
        for cls in range(num_classes):
            pred_mask = (preds_class_flat == cls)
            label_mask = (labels_flat == cls)
            
            intersection = np.logical_and(pred_mask, label_mask).sum()
            union = np.logical_or(pred_mask, label_mask).sum()
            
            if union > 0:
                ious.append(float(intersection / union))
        
        return float(np.mean(ious)) if ious else 0.0


def calculate_dice_score(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        all_preds = (all_preds > threshold).astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)
        
        all_preds_flat = all_preds.reshape(-1)
        all_labels_flat = all_labels.reshape(-1)
        
        intersection = np.logical_and(all_preds_flat, all_labels_flat).sum()
        dice = (2.0 * intersection) / (all_preds_flat.sum() + all_labels_flat.sum() + 1e-8)
        
        return float(dice)
    
    else:
        preds_class = np.argmax(all_preds, axis=1)
        
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        preds_class_flat = preds_class.reshape(-1)
        labels_flat = all_labels.reshape(-1)
        
        dice_scores = []
        for cls in range(num_classes):
            pred_mask = (preds_class_flat == cls)
            label_mask = (labels_flat == cls)
            
            intersection = np.logical_and(pred_mask, label_mask).sum()
            dice = (2.0 * intersection) / (pred_mask.sum() + label_mask.sum() + 1e-8)
            
            if pred_mask.sum() > 0 or label_mask.sum() > 0:
                dice_scores.append(float(dice))
        
        return float(np.mean(dice_scores)) if dice_scores else 0.0


def calculate_pixel_accuracy(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
        
        all_preds = (all_preds > threshold).astype(np.uint8)
        all_labels = all_labels.astype(np.uint8)
    else:
        all_preds = np.argmax(all_preds, axis=1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)
    
    all_preds_flat = all_preds.reshape(-1)
    all_labels_flat = all_labels.reshape(-1)
    
    correct = (all_preds_flat == all_labels_flat).sum()
    total = all_preds_flat.size
    
    return float(correct / total)


def calculate_precision(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)

        all_preds  = (all_preds > threshold).astype(np.uint8).reshape(-1)
        all_labels = all_labels.astype(np.uint8).reshape(-1)

        tp = np.logical_and(all_preds, all_labels).sum()
        fp = np.logical_and(all_preds, 1 - all_labels).sum()

        return float(tp / (tp + fp + 1e-8))

    else:
        preds_class = np.argmax(all_preds, axis=1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)

        preds_flat  = preds_class.reshape(-1)
        labels_flat = all_labels.reshape(-1)

        precisions = []
        for cls in range(num_classes):
            tp = np.logical_and(preds_flat == cls, labels_flat == cls).sum()
            fp = np.logical_and(preds_flat == cls, labels_flat != cls).sum()
            if tp + fp > 0:
                precisions.append(float(tp / (tp + fp + 1e-8)))

        return float(np.mean(precisions)) if precisions else 0.0


def calculate_recall(all_preds, all_labels, num_classes, threshold=0.5):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if num_classes == 1:
        if all_preds.ndim == 4:
            all_preds = all_preds.squeeze(1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)

        all_preds  = (all_preds > threshold).astype(np.uint8).reshape(-1)
        all_labels = all_labels.astype(np.uint8).reshape(-1)

        tp = np.logical_and(all_preds, all_labels).sum()
        fn = np.logical_and(1 - all_preds, all_labels).sum()

        return float(tp / (tp + fn + 1e-8))

    else:
        preds_class = np.argmax(all_preds, axis=1)
        if all_labels.ndim == 4:
            all_labels = all_labels.squeeze(1)

        preds_flat  = preds_class.reshape(-1)
        labels_flat = all_labels.reshape(-1)

        recalls = []
        for cls in range(num_classes):
            tp = np.logical_and(preds_flat == cls, labels_flat == cls).sum()
            fn = np.logical_and(preds_flat != cls, labels_flat == cls).sum()
            if tp + fn > 0:
                recalls.append(float(tp / (tp + fn + 1e-8)))

        return float(np.mean(recalls)) if recalls else 0.0


def calculate_model_complexity(model, input_size=(1, 3, 256, 256), device='cuda'):
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    dummy_input = torch.randn(input_size).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_allocated = 0.0
    
    gmacs = count_gmacs(model, input_size)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'memory_mb': memory_allocated,
        'gmacs': gmacs,            # value reported in the paper (1e9 multiply-accumulates)
        'gflops': 2 * gmacs        # 1 MAC = 2 floating-point operations
    }


def count_gmacs(model, input_size=(1, 3, 256, 256)):
    """
    Multiply-accumulate operations (in 1e9) counted with ptflops, which includes grouped,
    depthwise and padding='same' convolutions. Counting runs on a copy, so the model that is
    about to be trained is not modified. Returns NaN with a warning if counting is impossible.
    """
    try:
        from ptflops import get_model_complexity_info

        clone = copy.deepcopy(model).eval()
        # A2FPN also keeps its backbone as base_model / base_layers (not used in forward);
        # ptflops would count those layers twice.
        if type(clone).__name__ == 'A2FPN':
            del clone.base_model, clone.base_layers

        macs, _ = get_model_complexity_info(
            clone, tuple(input_size[1:]), as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
        if macs is None:
            raise RuntimeError('ptflops returned no result')
        return macs / 1e9
    except Exception as e:
        print(f"⚠️  GMACs could not be measured ({type(e).__name__}: {e}). "
              f"Install with: pip install ptflops==0.7.5")
        return float('nan')


def measure_inference_time(model, input_size=(1, 3, 256, 256), device='cuda', warmup=10, iterations=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    
    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000
    
    return {
        'avg_time_s': avg_time,
        'std_time_s': std_time,
        'min_time_s': min_time,
        'max_time_s': max_time,
        'fps': fps,
        'latency_ms': latency_ms
    }
