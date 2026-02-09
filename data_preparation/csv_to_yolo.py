import os 
import csv
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def csv_to_yolo(x,y,img_width,img_height,ball_size=8):
    x_center = x / img_width
    y_center = y / img_height
    
    width = ball_size / img_width
    height = ball_size / img_height
    
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    
    return x_center, y_center, width, height


def process_clip(clip_dir,output_images_dir,output_labels_dir,ball_size=8):
    clip_dir = Path(clip_dir)
    label_csv = clip_dir /"Label.csv"

    stats ={
        "total":0,
        "visible":0,
        "not_visible":0,
        "skipped":0
    }

    if not label_csv.exists():   
        print(f"No label file in clip {clip_dir}")
        return stats

    with open (label_csv,"r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"]+=1
            filename = row["file name"]
            visibility = int(row["visibility"])    
            image_path = clip_dir / filename
            if not image_path.exists():
                stats["skipped"]+=1
                continue

            try:
                img_width,img_height = get_image_size(image_path)
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
                stats["skipped"]+=1
                continue

            clip_name = f"{clip_dir.parent.name}_{clip_dir.name}"
            unique_filename = f"{clip_name}_{filename}"
            label_filename = unique_filename.replace(".jpg",".txt").replace(".png",".txt")
            
            output_image_path = Path(output_images_dir) / unique_filename
            shutil.copy2(image_path,output_image_path)


            output_label_path = Path(output_labels_dir)/ label_filename

            if visibility > 0:
                try: 
                    x = float(row["x-coordinate"])
                    y = float(row["y-coordinate"])
                    x_center,y_center,width,height = csv_to_yolo(x,y,img_width,img_height,ball_size)

                    with open(output_label_path,"w") as lf:
                        lf.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    stats["visible"] += 1

                except (ValueError,KeyError) as e:
                    # Missing or invalid coordinates
                    with open(output_label_path, 'w') as lf:
                        lf.write("")  # Empty label file (no objects)
                    stats['not_visible'] += 1
            else:
                with open(output_label_path,"w") as lf:
                    lf.write("")
                stats["not_visible"]+=1
    return stats

def find_all_clips(dataset_dir):
    clips =[]
    dataset_dir = Path(dataset_dir)
    for label_csv in dataset_dir.rglob("Label.csv"):
        clips.append(label_csv.parent)
    return sorted(clips)





def main():
    parser = argparse.ArgumentParser(
        description="Convert Tennis Ball Dataset CSV annotations to YOLO format"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/Users/hungcucu/Documents/usth/computer_vision/Tennis ball dataset/Dataset',
        help='Path to the Dataset directory containing game folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/Users/hungcucu/Documents/usth/computer_vision/tennis_analysis/yolo_dataset',
        help='Path to output YOLO format dataset'
    )
    parser.add_argument(
        '--ball_size',
        type=int,
        default=8,
        help='Estimated ball diameter in pixels (default: 12)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directories
    output_images_dir = output_dir / "images" / "all"
    output_labels_dir = output_dir / "labels" / "all"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ball size: {args.ball_size}px")
    print()
    
    # Find all clips
    clips = find_all_clips(input_dir)
    print(f"Found {len(clips)} clips to process")
    print()
    
    # Process all clips
    total_stats = {
        'total': 0,
        'visible': 0,
        'not_visible': 0,
        'skipped': 0
    }
    
    for clip_dir in tqdm(clips, desc="Processing clips"):
        stats = process_clip(
            clip_dir,
            output_images_dir,
            output_labels_dir,
            args.ball_size
        )
        
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Print summary
    print()
    print("=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    print(f"Total frames processed: {total_stats['total']}")
    print(f"Frames with visible ball: {total_stats['visible']}")
    print(f"Frames without visible ball: {total_stats['not_visible']}")
    print(f"Frames skipped (errors): {total_stats['skipped']}")
    print()
    print(f"Output images: {output_images_dir}")
    print(f"Output labels: {output_labels_dir}")
    print()
    print("Next step: Run split_dataset.py to create train/val/test splits")

if __name__ == "__main__":
    main()