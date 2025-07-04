import argparse
import timm
import torch
from timm.models import create_model
from huggingface_hub import ModelCard, ModelCardData, whoami

import model
import lsnet.model

MODEL_CONFIGS = {
    # M-series models (convolution + bilinear interpolation)
    'recnext_m0': {'series': 'M', 'embed_dim': (40, 80, 160, 320), 'depth': (2, 2, 9, 1), 'params': '2.5M', 'macs': '0.4G', 'latency': '1.0ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_m1': {'series': 'M', 'embed_dim': (48, 96, 192, 384), 'depth': (3, 3, 15, 2), 'params': '5.2M', 'macs': '0.9G', 'latency': '1.4ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_m2': {'series': 'M', 'embed_dim': (56, 112, 224, 448), 'depth': (3, 3, 15, 2), 'params': '6.8M', 'macs': '1.2G', 'latency': '1.5ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_m3': {'series': 'M', 'embed_dim': (64, 128, 256, 512), 'depth': (3, 3, 13, 2), 'params': '8.2M', 'macs': '1.4G', 'latency': '1.6ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_m4': {'series': 'M', 'embed_dim': (64, 128, 256, 512), 'depth': (5, 5, 25, 4), 'params': '14.1M', 'macs': '2.4G', 'latency': '2.4ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_m5': {'series': 'M', 'embed_dim': (80, 160, 320, 640), 'depth': (7, 7, 35, 2), 'params': '22.9M', 'macs': '4.7G', 'latency': '3.4ms', 'mlp_ratio': (2, 2, 2, 2)},
    
    # A-series models (linear attention + nearest interpolation)
    'recnext_a0': {'series': 'A', 'embed_dim': (40, 80, 160, 320), 'depth': (2, 2, 9, 1), 'params': '2.8M', 'macs': '0.4G', 'latency': '1.4ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_a1': {'series': 'A', 'embed_dim': (48, 96, 192, 384), 'depth': (3, 3, 15, 2), 'params': '5.9M', 'macs': '0.9G', 'latency': '1.9ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_a2': {'series': 'A', 'embed_dim': (56, 112, 224, 448), 'depth': (3, 3, 15, 2), 'params': '7.9M', 'macs': '1.2G', 'latency': '2.2ms', 'mlp_ratio': (2, 2, 2, 2)},
    'recnext_a3': {'series': 'A', 'embed_dim': (64, 128, 256, 512), 'depth': (3, 3, 13, 2), 'params': '9.0M', 'macs': '1.4G', 'latency': '2.4ms', 'mlp_ratio': (2, 2, 2, 1.875)},
    'recnext_a4': {'series': 'A', 'embed_dim': (64, 128, 256, 512), 'depth': (5, 5, 25, 4), 'params': '15.8M', 'macs': '2.4G', 'latency': '3.6ms', 'mlp_ratio': (2, 2, 2, 1.875)},
    'recnext_a5': {'series': 'A', 'embed_dim': (80, 160, 320, 640), 'depth': (7, 7, 35, 2), 'params': '25.7M', 'macs': '4.7G', 'latency': '5.6ms', 'mlp_ratio': (2, 2, 2, 1.875)},

    # L-series models (LSNet architecture)
    'recnext_t': {'series': 'L', 'embed_dim': (64, 128, 256, 512), 'depth': (0, 2, 8, 10), 'params': '12.1M', 'macs': '0.3G', 'latency': '1.8ms', 'mlp_ratio': (2, 2, 2, 1.5)},
    'recnext_s': {'series': 'L', 'embed_dim': (128, 256, 384, 512), 'depth': (0, 2, 8, 10), 'params': '15.8M', 'macs': '0.7G', 'latency': '2.0ms', 'mlp_ratio': (2, 2, 2, 1.5)},
    'recnext_b': {'series': 'L', 'embed_dim': (128, 256, 384, 512), 'depth': (2, 8, 8, 12), 'params': '19.3M', 'macs': '1.1G', 'latency': '2.5ms', 'mlp_ratio': (2, 2, 2, 1.5)},
}

def create_detailed_model_card(model_name, distillation=False):
    config = MODEL_CONFIGS.get(model_name, {})
    series = config.get('series', 'M')

    card_data = ModelCardData(
        language='en',
        license='apache-2.0',
        library_name='timm',
        tags=[
            'vision',
            'image-classification',
            'pytorch',
            'timm',
            'transformers'
        ],
        datasets=['imagenet-1k'],
        metrics=['accuracy'],
        model_name=model_name,
        pipeline_tag='image-classification'
    )

    card = ModelCard.from_template(
        card_data,
        template_path='model_card_template.md',
        model_name=model_name,
        series=series,
        distillation=distillation,
        config=config,
    )
    return card

def save_timm_to_hf(input_pth, model_name, distillation=False, publish=False):
    variant = 'dist' if distillation else 'base'
    print(f"Loading {variant} model from {input_pth}...")
    
    state_dict = torch.load(input_pth, map_location="cpu")['model']
    model = create_model(model_name, pretrained=False, distillation=distillation)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Creating detailed model card for {model_name}...")
    model_card = create_detailed_model_card(model_name, distillation)
    
    card_filename = f"pretrain/{model_name}_{variant}_model_card.md"
    model_card.save(card_filename)
    print(f"Model card saved to {card_filename}")
    
    if publish:
        repo_id = f"{whoami()['name']}/{model_name}.{variant}_300e_in1k"
        print(f"Pushing model and card to HuggingFace Hub: {repo_id}")
        timm.models.push_to_hf_hub(model, repo_id=repo_id, private=True)
        model_card.push_to_hub(repo_id)
        print("Model and detailed card uploaded successfully!")
    return model_card

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert timm model to Hugging Face format with detailed model card')
    parser.add_argument('--input_pth', type=str, required=True, help='Path to the input .pth model file')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the timm model (e.g., "recnext_m3")')
    parser.add_argument('--distillation', action='store_true', help='Whether to use distillation')
    parser.add_argument('--publish', action='store_true', help='Whether to publish the model to HuggingFace Hub')
    
    args = parser.parse_args()
    save_timm_to_hf(args.input_pth, args.model_name, args.distillation, args.publish)