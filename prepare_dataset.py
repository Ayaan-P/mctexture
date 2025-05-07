import os
import numpy as np
from PIL import Image
import json # To save label mapping

input_dir = 'minecraft_16x16_block_textures/'
output_file_data = 'minecraft_textures_dataset.npy'
output_file_labels = 'minecraft_textures_labels.npy'
output_file_label_map = 'minecraft_label_map.json'
image_size = (16, 16)
image_list = []
label_list = []
label_map = {}
current_label_id = 0

print(f"Reading images from {input_dir} and assigning labels...")

# Simple function to determine label based on filename
def get_label_from_filename(filename):
    name = os.path.splitext(filename)[0].lower()
    if 'log' in name or 'planks' in name or 'door' in name or 'sapling' in name or 'wood' in name:
        return 'wood'
    elif 'stone' in name or 'cobblestone' in name or 'brick' in name or 'sandstone' in name or 'hardened_clay' in name or 'prismarine' in name or 'purpur' in name or 'end_stone' in name or 'nether_brick' in name or 'red_nether_brick' in name:
        return 'stone/brick'
    elif 'dirt' in name or 'grass' in name or 'sand' in name or 'gravel' in name or 'clay' in name or 'mycelium' in name or 'soul_sand' in name or 'coarse_dirt' in name:
        return 'ground'
    elif 'ore' in name or 'block' in name or 'gold' in name or 'iron' in name or 'diamond' in name or 'emerald' in name or 'lapis' in name or 'coal' in name or 'redstone_block' in name:
         # Refine 'block' to exclude non-mineral blocks if possible, but keep it simple for now
         if 'command_block' in name or 'structure_block' in name:
             return 'utility' # Command/Structure blocks are more utility
         return 'mineral/block'
    elif 'glass' in name:
        return 'glass'
    elif 'wool' in name:
        return 'wool'
    elif 'flower' in name or 'plant' in name or 'tallgrass' in name or 'vine' in name or 'reeds' in name or 'deadbush' in name or 'waterlily' in name or 'fern' in name:
        return 'plant'
    elif 'tnt' in name:
        return 'explosive'
    elif 'ice' in name or 'snow' in name:
        return 'ice/snow'
    elif 'shulker' in name:
        return 'shulker'
    elif 'rail' in name or 'trip_wire' in name:
        return 'rail/wire'
    elif 'piston' in name or 'dispenser' in name or 'dropper' in name or 'comparator' in name or 'lever' in name or 'redstone_dust' in name or 'redstone_lamp' in name or 'torch' in name:
        return 'redstone/mechanism'
    elif 'cauldron' in name or 'brewing_stand' in name or 'enchanting_table' in name or 'crafting_table' in name or 'furnace' in name or 'anvil' in name or 'item_frame' in name or 'ladder' in name or 'painting' in name:
        return 'utility'
    elif 'slime' in name:
        return 'slime'
    elif 'mob_spawner' in name:
        return 'mob_spawner'
    elif 'cake' in name:
        return 'food'
    elif 'melon' in name or 'pumpkin' in name:
        return 'plant/food'
    else:
        return 'other'


# List all files in the input directory
try:
    files = os.listdir(input_dir)
except FileNotFoundError:
    print(f"Error: Directory '{input_dir}' not found.")
    exit()

# Filter for PNG files and process them
png_files = [f for f in files if f.endswith('.png')]

if not png_files:
    print(f"No PNG files found in '{input_dir}'.")
    exit()

for filename in png_files:
    file_path = os.path.join(input_dir, filename)
    try:
        img = Image.open(file_path).convert('RGB') # Convert to RGB to ensure consistent channel count
        if img.size != image_size:
            print(f"Warning: Skipping '{filename}' due to incorrect size {img.size}. Expected {image_size}.")
            continue

        # Get label and assign ID
        label_str = get_label_from_filename(filename)
        if label_str not in label_map:
            label_map[label_str] = current_label_id
            current_label_id += 1
        label_id = label_map[label_str]

        img_array = np.array(img)
        image_list.append(img_array)
        label_list.append(label_id)
        print(f"Processed {filename} with label '{label_str}' (ID: {label_id})")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

if not image_list:
    print("No valid images were processed. Dataset not created.")
else:
    # Stack all image arrays into a single NumPy array
    dataset = np.stack(image_list, axis=0)
    labels = np.array(label_list, dtype=np.int64) # Use int64 for labels

    # Save the dataset and labels
    try:
        np.save(output_file_data, dataset)
        np.save(output_file_labels, labels)
        with open(output_file_label_map, 'w') as f:
            json.dump(label_map, f, indent=4)
        print(f"Successfully created dataset: {output_file_data} with shape {dataset.shape}")
        print(f"Successfully created labels: {output_file_labels} with shape {labels.shape}")
        print(f"Successfully created label map: {output_file_label_map}")
    except Exception as e:
        print(f"Error saving dataset or labels: {e}")
