import os 
import random 
import shutil 
import argparse
from tqdm import tqdm 
from pathlib import Path


def split_dataset(input_dir, train_ratio = 0.8, val_ratio = 0.1, test_ratio =0.1, seed = 42):
    random.seed(seed)
    input_dir = Path(input_dir)
    images_all = input_dir / "images" / "all"
    labels_all = input_dir / "labels" / "all"

    if not images_all.exists():
        raise FileExistsError(f"Images directory not found:{images_all}")

    if not labels_all.exists():
        raise FileExistsError(f"Label directory not found :{labels_all}")


    image_files = list(Path(images_all).glob("*.jpg")) + list(Path(images_all).glob("*.[png]"))
    print(f"Found {len(image_files)} images")        


    random.shuffle(image_files)

    total = len(image_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    print(f"Split: train={train_size}, val={val_size}, test={test_size}")

    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size+val_size:]


    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        split_images_dir = input_dir/"images"/split_name
        split_labels_dir = input_dir/"labels"/split_name
        split_images_dir.mkdir(parents = True, exist_ok = True)
        split_labels_dir.mkdir(parents = True, exist_ok = True)
        
        for img_path in tqdm(files, desc = split_name):
            dst_img = split_images_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            
            label_name = img_path.stem +".txt"
            src_label = labels_all / label_name
            dst_label = split_labels_dir / label_name

            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                dst_label.touch()

    print(f"Train: {train_size} images ({train_ratio*100:.0f}%)")
    print(f"Val: {val_size} images ({val_ratio*100:.0f}%)")
    print(f"Test: {test_size} images ({test_ratio*100:.0f}%)")
    print(f"\nConfig file: {input_dir / 'tennis_ball.yaml'}")

def create_yaml_config(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    yaml_content =f"""path :{dataset_dir}
train: images/train
val:images/val
test:images/test

names:
    0:ball


nc:1
"""

    yaml_path = dataset_dir/"tennis_ball.yaml"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"created yaml file :{yaml_path}")

def count_labels_with_objects(labels_dir):
    labels_dir  = Path(labels_dir)
    total = 0
    with_objects = 0

    for label_file in labels_dir.glob("*.txt"):
        total +=1 
        if label_file.stat().st_size >0:
            with_objects +=1

    return with_objects, total

def print_stats(dataset_dir):
    dataset_dir = Path(dataset_dir)
    
    for split in ["train", "val", "test"]:
        labels_dir = dataset_dir / "labels" / split
        if labels_dir.exists():
            with_obj, total = count_labels_with_objects(labels_dir)
            pct = (with_obj / total * 100) if total > 0 else 0
            print(f"{split:5s}: {total:5d} images, {with_obj:5d} with ball ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val/test sets"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/yolo_dataset',
        help='Path to YOLO dataset directory'
    )
    parser.add_argument(
        '--train',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--stats_only',
        action='store_true',
        help='Only print statistics, do not split'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0")
    
    if args.stats_only:
        print_stats(args.input_dir)
    else:
        split_dataset(
            args.input_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )
        print_stats(args.input_dir)
        create_yaml_config(args.input_dir)

if __name__ == "__main__":
    main()

