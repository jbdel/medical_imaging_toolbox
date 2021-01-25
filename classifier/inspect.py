import argparse, os, random
import torch
import json
import pprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    metrics = ckpt["metrics"]
    # metrics = json.dumps(ckpt["metrics"])
    # print(pprint.pprint(metrics)
    #       )

    print(metrics['classification_report'])