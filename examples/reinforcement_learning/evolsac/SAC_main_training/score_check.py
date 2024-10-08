import csv
import os


def extract_score_from_file(file_path):
    """Extract the last value (RealAI Score) from the scores.csv file."""
    with open(file_path, mode="r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        row = next(csv_reader)  # Read the actual data
        return float(row[-1])  # Return the last value as the score


def find_highest_score(root_directory):
    """Find the highest score and the corresponding directory."""
    highest_score = float("-inf")
    best_directory = None

    # Iterate through directories
    for dir_name in os.listdir(root_directory):
        dir_path = os.path.join(root_directory, dir_name)

        # Check if it's a directory
        if os.path.isdir(dir_path):
            file_path = os.path.join(dir_path, "scores.csv")

            # Check if the scores.csv file exists in the directory
            if os.path.exists(file_path):
                try:
                    score = extract_score_from_file(file_path)

                    # Check if this score is the highest
                    if score > highest_score:
                        highest_score = score
                        best_directory = dir_name

                    elif score > 0.45:
                        print(f"Directory {dir_name} has score = {score}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return highest_score, best_directory


# Specify the root directory where the numbered folders are located
root_directory = "./data_acrobot/training_noisy.py-3.0-0.9-0-0-0.01/evolsac/"
# Find the highest score and the directory containing it
highest_score, best_directory = find_highest_score(root_directory)

if best_directory is not None:
    print(f"The highest score is {highest_score}, found in directory: {best_directory}")
else:
    print("No valid scores found.")
