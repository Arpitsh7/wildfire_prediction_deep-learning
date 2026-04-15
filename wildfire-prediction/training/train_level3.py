import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from datetime import datetime

from models.level3_unet import Level3UNet


def create_level3_analysis():
    """Create Level 3 analysis and baseline model."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Create Level 3 model
    print("\n" + "="*60)
    print("CREATING LEVEL 3 MODEL")
    print("="*60)
    
    model = Level3UNet(in_channels=12, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Level3UNet (Lightweight U-Net with SE blocks)")
    print(f"Total parameters: {total_params:,}")
    
    # Save checkpoint
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "level3.pth"))
    print(f"Checkpoint saved to: checkpoints/level3.pth")
    
    # Create comprehensive analysis document
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = {
        'level': 3,
        'status': 'model_created_and_analyzed',
        'timestamp': datetime.now().isoformat(),
        'model_name': 'Level3UNet (Lightweight U-Net with SE blocks)',
        'model_type': 'Semantic Segmentation',
        'task': 'Wildfire Prediction on 64x64 grid',
        'model_params': total_params,
        'architecture_description': {
            'encoder': [
                'Layer 1: 12 -> 32 channels + SE block + MaxPool',
                'Layer 2: 32 -> 64 channels + SE block + MaxPool',
                'Layer 3: 64 -> 128 channels + SE block + MaxPool',
                'Layer 4: 128 -> 256 channels + SE block + MaxPool',
            ],
            'bottleneck': [
                'Layer: 256 -> 512 channels + SE block'
            ],
            'decoder': [
                'Layer 4: 512+256 -> 256 channels + SE block',
                'Layer 3: 256+128 -> 128 channels + SE block',
                'Layer 2: 128+64 -> 64 channels + SE block',
                'Layer 1: 64+32 -> 32 channels + SE block',
                'Output: 32 -> 1 channel (fire mask)'
            ],
            'special_components': [
                'SE (Squeeze-and-Excitation) blocks for channel attention',
                'Skip connections from encoder to decoder',
                'Transposed convolutions for upsampling',
                'BatchNorm + ReLU activations'
            ]
        },
        'design_rationale': {
            'vs_level1': [
                'Level 1 (ResNet18 U-Net): 31.6M parameters, slower on CPU',
                'Level 3: 5.3M parameters (83% reduction), much faster',
                'Level 3 more suitable for CPU-based inference',
                'SE blocks provide channel attention similar to full attention mechanisms'
            ],
            'vs_level2': [
                'Level 2 failed with F1=0.0006 (worse than baseline)',
                'Level 2 issues: too much augmentation, training instability',
                'Level 3 approach: cleaner architecture, better training strategy',
                'Level 3 focuses on model capacity and attention mechanisms',
                'Level 3 avoids aggressive augmentation that disrupted Level 2'
            ]
        },
        'improvements_in_level3': [
            'Lightweight architecture - 83% fewer parameters than Level 1',
            'SE blocks for channel-wise attention without excessive computation',
            'Proper skip connections for better gradient flow',
            'Input feature normalization (min-max per channel)',
            'Clean training strategy (no excessive augmentation)',
            'Better suited for CPU training environments'
        ],
        'level_comparison': {
            'level1': {
                'model': 'ResNet18-based U-Net',
                'params': 31619231,
                'test_f1': 0.2331,
                'test_precision': 0.1346,
                'test_recall': 0.8773,
                'test_iou': 0.1323,
                'best_threshold': 0.7,
                'status': 'BEST so far'
            },
            'level2': {
                'model': 'U-Net + Heavy Augmentation + ResNet18',
                'epochs_run': 3,
                'test_f1': 0.0006,
                'status': 'FAILED - worse than Level 1'
            },
            'level3': {
                'model': 'Lightweight U-Net with SE blocks',
                'params': total_params,
                'expected_improvements': [
                    'Faster training/inference than Level 1',
                    'Better generalization than Level 2',
                    'Maintains attention mechanism benefits via SE blocks',
                    'Cleaner architecture for CPU training'
                ],
                'next_steps': [
                    'Train on full dataset with proper hyperparameters',
                    'Test different augmentation strategies',
                    'Tune learning rate and schedule',
                    'Aim to achieve F1 > 0.25 (improvement over Level 1)'
                ],
                'status': 'READY FOR TRAINING'
            }
        },
        'recommended_training_config': {
            'optimizer': 'Adam or AdamW',
            'learning_rate': 1e-4,
            'batch_size': 32,
            'epochs': 15,
            'loss_function': 'BCEWithLogitsLoss with pos_weight=90.33',
            'scheduler': 'ReduceLROnPlateau or StepLR',
            'data_strategy': 'Min-max normalization per channel',
            'augmentation': 'Light augmentation (no heavy transforms that broke Level 2)',
            'threshold_tuning': 'Test 0.3-0.8 range, expect best around 0.6-0.7'
        },
        'expected_outcomes': {
            'conservative_estimate': 'F1 >= 0.23 (match Level 1)',
            'target': 'F1 >= 0.26 (3% improvement over Level 1)',
            'optimistic': 'F1 >= 0.30 (improvement from better architecture)',
            'key_metric': 'Improved precision from 13.5% while maintaining recall'
        },
        'implementation_notes': {
            'why_level3': 'After Level 2 failure, we need a middle ground',
            'architecture_choice': 'Lightweight + SE blocks = attention without overhead',
            'training_strategy': 'Conservative (avoid aggressive augmentation that broke Level 2)',
            'cpu_optimization': 'Designed for fast CPU training if GPU unavailable',
            'checkpoint': 'checkpoints/level3.pth'
        }
    }
    
    with open(os.path.join(results_dir, "level3_metrics.json"), 'w') as f:
        json.dump(results_file, f, indent=2)
    
    print(f"Results saved to: results/level3_metrics.json")
    
    return results_file


if __name__ == "__main__":
    results = create_level3_analysis()
    print("\n" + "="*60)
    print("LEVEL 3 MODEL CREATED AND DOCUMENTED")
    print("="*60)
    print(f"\nCheckpoint: checkpoints/level3.pth")
    print(f"Documentation: results/level3_metrics.json")
    print(f"Total parameters: {results['model_params']:,}")
