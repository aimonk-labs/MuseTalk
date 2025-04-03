from PIL import Image
import imagehash
import os

def find_similar_images(folder_path):
    """Find duplicate or near-duplicate images using perceptual hashing."""
    hashes = {}  # Store hash values and corresponding image paths
    duplicates = []

    for filename in sorted(os.listdir(folder_path)):  # Sort for consistency
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            img = Image.open(file_path)
            img_hash = str(imagehash.average_hash(img))  # Generate perceptual hash

            if img_hash in hashes:
                print(f"Visually similar images: {file_path} == {hashes[img_hash]}")
                duplicates.append((file_path, hashes[img_hash]))
            else:
                hashes[img_hash] = file_path

    return duplicates

# Example usage
folder = "/vidgen/VIDGEN_AI_CMPT_MODAL/vidgen_ai_cmpt/models/vidgen_editor/1/IP_LAP_temp_results/20250320-115423-6f4bd8958fe944c4b0032cbe31677c31/musetalk_matt_file_20250320-115423-ef78f0db43644a69a42f058e7429ebde/matts"
duplicates = find_similar_images(folder)
if duplicates:
    print("\n✅ Found visually similar images!")
else:
    print("\n✅ No duplicates found.")
