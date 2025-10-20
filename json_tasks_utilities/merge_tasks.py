import json
import os

def merge_task_files(file_paths, output_file):
    """
    Merges multiple Label Studio task JSON files into a single file.

    Args:
        file_paths (list): A list of paths to the input task.json files.
        output_file (str): The path for the merged output JSON file.
    """
    merged_tasks = []
    total_task_count = 0

    print("🚀 Starting merge process...")

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
                # Ensure the loaded data is a list
                if isinstance(tasks, list):
                    print(f"✅ Reading {len(tasks)} tasks from '{file_path}'...")
                    merged_tasks.extend(tasks)
                    total_task_count += len(tasks)
                else:
                    print(f"⚠️ Warning: Expected a list of tasks in '{file_path}', but found {type(tasks)}. Skipping.")

        except FileNotFoundError:
            print(f"❌ Error: The file '{file_path}' was not found. Skipping.")
        except json.JSONDecodeError:
            print(f"❌ Error: Could not decode JSON from '{file_path}'. Please ensure it's a valid JSON file.")
        except Exception as e:
            print(f"An unexpected error occurred with '{file_path}': {e}")
            
    if not merged_tasks:
        print("No tasks were found to merge. Exiting.")
        return

    # Write the combined list of tasks to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=4 makes the output file human-readable
            json.dump(merged_tasks, f, indent=4)
        print(f"\n✨ Success! Merged a total of {total_task_count} tasks into '{output_file}'.")
    except Exception as e:
        print(f"\n❌ Error: Could not write to the output file '{output_file}': {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # 1. List the JSON files you want to merge
    input_files = ['task1.json', 'task2.json']

    # 2. Name the final merged file
    output_filename = 'merged_tasks.json'
    # -------------------

    # Check if input files exist
    existing_files = [f for f in input_files if os.path.exists(f)]
    if len(existing_files) != len(input_files):
        print("Warning: One or more input files were not found. Proceeding with the files that were found.")

    if not existing_files:
        print("Error: No input files found. Please check your configuration.")
    else:
        merge_task_files(existing_files, output_filename)