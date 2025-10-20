import json
import os

def count_unique_images(file_paths):
    """
    Counts the number of unique images across multiple Label Studio task JSON files.

    Args:
        file_paths (list): A list of strings, where each string is the path to a task.json file.

    Returns:
        int: The total count of unique images.
    """
    unique_image_paths = set()
    total_image_paths = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
                
                # Iterate through each task in the JSON file
                for task in tasks:
                    # The image path is nested under the 'data' key
                    if 'data' in task and 'image' in task['data']:
                        image_path = task['data']['image']
                        unique_image_paths.add(image_path)
                        total_image_paths.append(image_path)

        except FileNotFoundError:
            print(f"⚠️ Error: The file '{file_path}' was not found. Skipping.")
        except json.JSONDecodeError:
            print(f"️⚠️ Error: The file '{file_path}' is not a valid JSON file. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred with file '{file_path}': {e}")
            
    return len(unique_image_paths), len(total_image_paths)

if __name__ == "__main__":
    # --- Configuration ---
    # Add the paths to your two JSON files here
    json_file_1 = 'task.json'
    # json_file_2 = 'task2.json'
    # -------------------

    files_to_process = [json_file_1]
    
    # Check if the specified files exist before processing
    existing_files = [f for f in files_to_process if os.path.exists(f)]
    
    if not existing_files:
        print("❌ Error: None of the specified JSON files were found. Please check the file names and paths.")
    else:
        total_unique_images, total_images = count_unique_images(existing_files)
        print(f"✨ Found a total of {total_unique_images} unique images out of {total_images} images, across the provided files.")