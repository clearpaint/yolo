import os
import torch
from ultralytics import YOLO
import shutil
import tempfile
import yaml

def convert_polygon_to_bbox(original_label_path, converted_label_path):
    """
    Converts polygon labels to bounding box labels in YOLO format.

    Args:
        original_label_path (str): Path to the original label file with polygon coordinates.
        converted_label_path (str): Path to save the converted label file with bounding boxes.
    """
    with open(original_label_path, 'r') as infile, open(converted_label_path, 'w') as outfile:
        for line_num, line in enumerate(infile, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Warning: Line {line_num} in '{original_label_path}' does not have enough data. Skipping.")
                continue  # Not enough data to form a bounding box
            class_id = parts[0]
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                print(f"Warning: Line {line_num} in '{original_label_path}' has an odd number of coordinates. Skipping.")
                continue  # Coordinates should be in pairs
            xs = coords[::2]
            ys = coords[1::2]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            # Ensure normalized values are within [0,1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            outfile.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def prepare_converted_dataset(original_dataset_dir, converted_dataset_dir):
    """
    Prepares a converted dataset directory with bounding box labels.

    Args:
        original_dataset_dir (str): Path to the original dataset directory (train, val, test).
        converted_dataset_dir (str): Path to the converted dataset directory.
    """
    images_dir = os.path.join(original_dataset_dir, 'images')
    labels_dir = os.path.join(original_dataset_dir, 'labels')
    converted_labels_dir = os.path.join(converted_dataset_dir, 'labels')

    if not os.path.exists(images_dir):
        print(f"❌ Images directory '{images_dir}' does not exist. Skipping this split.")
        return

    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory '{labels_dir}' does not exist. Skipping this split.")
        return

    os.makedirs(converted_labels_dir, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"🔄 Converting {len(label_files)} label files in '{labels_dir}'.")

    for filename in label_files:
        original_label_path = os.path.join(labels_dir, filename)
        converted_label_path = os.path.join(converted_labels_dir, filename)
        convert_polygon_to_bbox(original_label_path, converted_label_path)

    # Copy images to the converted dataset directory
    converted_images_dir = os.path.join(converted_dataset_dir, 'images')
    os.makedirs(converted_images_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🔄 Copying {len(image_files)} images from '{images_dir}' to '{converted_images_dir}'.")
    for filename in image_files:
        original_image_path = os.path.join(images_dir, filename)
        converted_image_path = os.path.join(converted_images_dir, filename)
        shutil.copy2(original_image_path, converted_image_path)

def create_temp_converted_dataset(project_dir, data_yaml_path):
    """
    Creates a temporary converted dataset directory with bounding box labels.

    Args:
        project_dir (str): Path to the project directory containing data.yaml.
        data_yaml_path (str): Path to the original data.yaml.

    Returns:
        str: Path to the temporary converted dataset directory.
    """
    with open(data_yaml_path, 'r') as infile:
        data = yaml.safe_load(infile)

    temp_dir = tempfile.mkdtemp(prefix="yolo_converted_")
    print(f"✅ Temporary converted dataset directory created at '{temp_dir}'.")

    for split in ['train', 'val', 'test']:
        if split in data:
            split_path = data[split]
            # Determine the base directory for the split
            split_dir = os.path.dirname(split_path)  # e.g., 'train'
            original_split_dir = split_dir if os.path.isabs(split_dir) else os.path.join(project_dir, split_dir)
            converted_split_dir = os.path.join(temp_dir, split)
            print(f"🔄 Preparing split '{split}' from '{original_split_dir}' to '{converted_split_dir}'.")
            prepare_converted_dataset(original_split_dir, converted_split_dir)
        else:
            print(f"⚠️ Split '{split}' not found in 'data.yaml'. Skipping.")

    return temp_dir

def create_temp_data_yaml(original_data_yaml, converted_dataset_dir, temp_data_yaml_path):
    """
    Creates a temporary data.yaml pointing to the converted dataset.

    Args:
        original_data_yaml (str): Path to the original data.yaml.
        converted_dataset_dir (str): Path to the converted dataset directory.
        temp_data_yaml_path (str): Path to save the temporary data.yaml.
    """
    with open(original_data_yaml, 'r') as infile:
        data = yaml.safe_load(infile)

    # Update paths to point to the converted dataset
    for split in ['train', 'val', 'test']:
        if split in data:
            data[split] = os.path.join(converted_dataset_dir, split, 'images')

    # Remove Roboflow section if present
    if 'roboflow' in data:
        del data['roboflow']

    with open(temp_data_yaml_path, 'w') as outfile:
        yaml.dump(data, outfile)

    print(f"✅ Temporary data.yaml created at '{temp_data_yaml_path}'.")

def train_yolo_model():
    """
    Trains a YOLO model using the Ultralytics library.
    Utilizes GPU if available and optimizes memory and multithreading.
    Includes console outputs for monitoring training progress.
    Converts polygon labels to bounding boxes on-the-fly using a temporary dataset.
    """
    # Determine the device to use (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚙️ Using device: {device}")

    # Initialize the YOLO model using a pre-defined architecture
    model = YOLO('yolov8n.pt')  # Change to 'yolov8s.pt', etc., if desired
    print(f"✅ YOLO model 'yolov8n.pt' loaded successfully.")

    # Determine the number of CPU cores for data loading
    num_workers = os.cpu_count() if os.cpu_count() else 4
    print(f"🔧 Using {num_workers} data loader workers for multithreading.")

    # Path to the original data.yaml
    original_data_yaml = 'data.yaml'

    # Check if the data.yaml file exists
    if not os.path.exists(original_data_yaml):
        print(f"❌ Data configuration file '{original_data_yaml}' not found. Please provide a valid data.yaml file.")
        return

    # Create a temporary converted dataset
    print("🔄 Converting polygon labels to bounding boxes...")
    project_dir = os.path.dirname(os.path.abspath(original_data_yaml))
    converted_dataset_dir = create_temp_converted_dataset(project_dir, original_data_yaml)
    print(f"✅ Converted dataset created at '{converted_dataset_dir}'.")

    # Create a temporary data.yaml pointing to the converted dataset
    temp_data_yaml = os.path.join(converted_dataset_dir, 'data_temp.yaml')
    create_temp_data_yaml(original_data_yaml, converted_dataset_dir, temp_data_yaml)
    print(f"✅ Temporary data.yaml created at '{temp_data_yaml}'.")

    # Verify that converted_dataset_dir/val/images exists and contains images
    val_images_path = os.path.join(converted_dataset_dir, 'val', 'images')
    if not os.path.exists(val_images_path):
        print(f"❌ Validation images directory '{val_images_path}' does not exist.")
        print("❌ Please ensure that your original 'valid/images' directory contains images.")
        # Cleanup
        print("🧹 Cleaning up temporary files...")
        shutil.rmtree(converted_dataset_dir)
        print("✅ Cleanup completed.")
        return
    elif not os.listdir(val_images_path):
        print(f"❌ Validation images directory '{val_images_path}' is empty.")
        print("❌ Please ensure that your original 'valid/images' directory contains images.")
        # Cleanup
        print("🧹 Cleaning up temporary files...")
        shutil.rmtree(converted_dataset_dir)
        print("✅ Cleanup completed.")
        return
    else:
        print(f"✅ Validation images found at '{val_images_path}'.")

    # Define training parameters
    training_params = {
        'data': temp_data_yaml,                # Path to the temporary data.yaml
        'epochs': 100,                          # Number of training epochs
        'batch': 16,                            # Batch size (adjust based on GPU memory)
        'device': device,                       # Device to train on
        'workers': num_workers,                 # Number of data loader workers
        'cache': True,                          # Cache images for faster training
        'multi_scale': True,                    # Enables multi-scale training
        'optimizer': 'Adam',                    # Optimizer (can be 'SGD', 'Adam', etc.)
        'project': 'runs/train',                # Directory to save training runs
        'name': 'yolov8_custom',                # Name of the training run
        'exist_ok': True,                       # Overwrite if the project/name exists
        'save': True,                           # Save the best model
        'save_period': -1,                      # Save only the final model
        'verbose': True,                        # Enable verbose logging
        'resume': False,                        # Resume training from last checkpoint
        # 'callbacks': [progress_callback],     # Removed as it's unsupported
        # Additional parameters can be added here
    }

    # Start training
    print("\n🚀 Starting training...")
    try:
        model.train(**training_params)
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
    finally:
        # Clean up the temporary dataset directory after training
        print("🧹 Cleaning up temporary files...")
        shutil.rmtree(converted_dataset_dir)
        print("✅ Cleanup completed.")
    print("🏁 Training process has been completed.")

if __name__ == "__main__":
    train_yolo_model()
