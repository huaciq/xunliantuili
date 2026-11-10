"""Train official INP-Former components with image-level good/bad labels."""

import argparse
import csv
import json
import math
import os
from functools import partial
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--data", type=Path, default=Path("/workspace/dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("PLATFORM_OUTPUT_DIR", "/workspace/output")),
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.input_size < 56 or args.input_size % 14:
        parser.error(
            "epochs and batch size must be positive; input size must be >= 56 and divisible by 14"
        )
    return args


def main():
    args = parse_args()
    import torch
    from dataset import get_data_transforms
    from models import vit_encoder
    from models.uad import INP_Former
    from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block
    from optimizers import StableAdamW
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from utils import (
        WarmCosineScheduler,
        cal_anomaly_maps,
        get_gaussian_kernel,
        global_cosine_hm_adaptive,
        setup_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This job requires a CUDA GPU")
    weight = Path(__file__).parent / "backbones/weights/dinov2_vitb14_reg4_pretrain.pth"
    if not weight.is_file() or not weight.stat().st_size:
        raise FileNotFoundError(f"Bundle the DINOv2 encoder weight at {weight}")
    setup_seed(1)
    device = torch.device("cuda:0")
    transform, _ = get_data_transforms(args.input_size, args.input_size)
    train = ImageFolder(args.data / "train", transform=transform)
    test = ImageFolder(args.data / "test", transform=transform)
    if train.classes != ["good"] or set(test.classes) != {"good", "bad"}:
        raise ValueError("Expected train/good, test/good and test/bad with images")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(
        f"GPU: {torch.cuda.get_device_name(device)}; train={len(train)}; test={len(test)}",
        flush=True,
    )

    encoder_name = "dinov2reg_vit_base_14"
    encoder = vit_encoder.load(encoder_name).requires_grad_(False)
    width = encoder.embed_dim
    block_options = {
        "dim": width,
        "num_heads": 12,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-8),
    }
    bottleneck = nn.ModuleList([Mlp(width, width * 4, width, drop=0)])
    aggregation = nn.ModuleList([Aggregation_Block(**block_options)])
    decoder = nn.ModuleList(Prototype_Block(**block_options) for _ in range(8))
    prototypes = nn.ParameterList([nn.Parameter(torch.randn(6, width))])
    groups = [list(range(4)), list(range(4, 8))]
    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=aggregation,
        decoder=decoder,
        target_layers=list(range(2, 10)),
        remove_class_token=True,
        fuse_layer_encoder=groups,
        fuse_layer_decoder=groups,
        prototype_token=prototypes,
    ).to(device)
    trainable = nn.ModuleList([bottleneck, aggregation, decoder, prototypes])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    optimizer = StableAdamW(
        trainable.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10,
    )
    steps = args.epochs * len(train_loader)
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=1e-3,
        final_value=1e-4,
        total_iters=steps,
        warmup_iters=min(100, steps // 10),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "train/loss"))
        for epoch in range(1, args.epochs + 1):
            model.train()
            encoder.eval()
            loss_total = 0.0
            for batch, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                features, reconstruction, penalty = model(batch.to(device))
                loss = global_cosine_hm_adaptive(features, reconstruction, y=3) + 0.2 * penalty
                value = loss.item()
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite training loss in epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                loss_total += value
            mean_loss = loss_total / len(train_loader)
            writer.writerow((epoch, mean_loss))
            stream.flush()
            print(f"Epoch {epoch}/{args.epochs}: loss={mean_loss:.6f}", flush=True)

    model.eval()
    scores = []
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for batch, _ in test_loader:
            features, reconstruction, _ = model(batch.to(device))
            anomaly, _ = cal_anomaly_maps(features, reconstruction, args.input_size)
            pixels = gaussian(
                F.interpolate(anomaly, size=(256, 256), mode="bilinear", align_corners=False)
            ).flatten(1)
            k = max(1, pixels.size(1) // 100)
            scores.extend(pixels.topk(k, dim=1).values.mean(dim=1).cpu().tolist())

    labels = [int(label == test.class_to_idx["bad"]) for _, label in test.samples]
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "is_bad", "anomaly_score"))
        for (filename, _), label, score in zip(test.samples, labels, scores):
            writer.writerow((Path(filename).relative_to(args.data).as_posix(), label, score))
    report = {
        "image_auroc": roc_auc_score(labels, scores),
        "image_average_precision": average_precision_score(labels, scores),
        "train_good": len(train),
        "test_good": len(test) - sum(labels),
        "test_bad": sum(labels),
        "encoder": encoder_name,
        "inp_num": 6,
        "input_size": args.input_size,
        "crop_size": args.input_size,
        "epochs": args.epochs,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.output / "model.pth")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
