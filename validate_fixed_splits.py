"""Validate dataset/raw/{train,validation,test} before training or evaluation."""

import argparse

from utils import FIXED_SPLITS, validate_fixed_split_coverage


def main():
    parser = argparse.ArgumentParser(description="고정 데이터 split 완전성/누수 검사")
    parser.add_argument("--data-dir", default="dataset/raw")
    args = parser.parse_args()

    counts = validate_fixed_split_coverage(args.data_dir)
    print("[fixed split validation passed]")
    for split in FIXED_SPLITS:
        print(f" - {split}: {counts[split]} frames")


if __name__ == "__main__":
    main()
