#!/usr/bin/env python3
"""
UR Fall Detection (URFD), SisFall, UP-Fall quick fetch helper.
Review licenses before download/use. Update URLs as needed.
"""
import argparse, os, sys, urllib.request

DATASETS = {
    "urfd": [
        # Example link (update if mirror/host changes)
        "https://www.verlab.dcc.ufmg.br/geral-research/falldetection-dataset/URFD/URFD.rar"
    ],
    "sisfall": [
        "https://sisfall-dataset.github.io/files/SisFall.zip"
    ],
    "upfall": [
        "https://zenodo.org/record/4278415/files/UP-Fall%20Detection%20Dataset.zip?download=1"
    ]
}

def download(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    name = url.split("/")[-1].split("?")[0]
    out = os.path.join(dest_dir, name)
    print(f"Downloading {url} -> {out}")
    urllib.request.urlretrieve(url, out)
    print("Done.")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw/urfd")
    ap.add_argument("--dataset", choices=list(DATASETS.keys())+["all"], default="urfd")
    args = ap.parse_args()
    if args.dataset == "all":
        for k in DATASETS:
            for url in DATASETS[k]:
                download(url, args.dest)
    else:
        for url in DATASETS[args.dataset]:
            download(url, args.dest)

if __name__ == "__main__":
    main()
