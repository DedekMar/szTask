from joblib import load
import sys
import argparse

import pandas as pd


# Function to try and load the model from pkl file given the model_file path parameter

def load_model(model_file):
    try:
        model = load(model_file)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

# Function to try and load the data from excel file given the data_file path parameter
# Probably wont be loaded through excel files but rather through some DBs I'd assume.
def load_data(data_file):
    try:
        new_batch = pd.read_excel(data_file, engine='openpyxl')
        print("Data loaded successfully.")
        return new_batch
    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        exit(1)
    except Exception as e:
        print(f"Error loading the data: {e}")
        exit(1)


# A demo of a potential automatization of this script. Of course ideally it'd get split into more functions 
# The new_batch.xlsx file here is just the original file with the target column removed to simulate a "new batch" for the purpose of this demo script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and use a trained model for predictions.")
    
    # Add command-line arguments for model file and data file
    parser.add_argument("-model", default = "model.pkl", help="Path to the model file (in .pkl format)")
    parser.add_argument("-data", default = "new_batch.xlsx", help="Path to the input data file (in .xlsx format)")

    args = parser.parse_args()

    model_file_location = args.model
    model = load_model(model_file_location)

    data_file_location = args.data
    new_batch = load_data(data_file_location)

    # Assuming the new batch contains both old and new players and the flags are available
    # This new player batch must contain the same attributes in the same format like they were used for model training
    new_players = new_batch[new_batch["NEW_PLAYER_FLAG"] == 1]

    # Preprocess the data so that it contains the same features that were used for the model training
    dropped_cols = ["PLAYER_ID", "NEW_PLAYER_FLAG", "DATA_SOURCE_SYSTEM", "TARGET_GROUP_FLAG", "COUNT_LIVE_BET"]
    all_cols = new_players.columns 
    attribute_cols = [x for x in all_cols if x not in dropped_cols]
    selected_df = new_players[attribute_cols]

    # Now nan values need to be handled. This again is up for discussion if they should be replaced with mean or other methods or just dropped as there are very very few like in the previous dataset

    selected_df  = selected_df.dropna()

    # Classify the new players

    predictions = model.predict(selected_df)

    selected_df["TARGET_GROUP_PREDICTION"] = predictions

    targeted_players = selected_df[selected_df["TARGET_GROUP_PREDICTION"] == 1]

    # Return the printout of positive classifications as output...
    
    positive_counts = (selected_df["TARGET_GROUP_PREDICTION"] == 1).sum()
    print(f"Number of new players classified {positive_counts}")

    # Return / load / unload whatever is needed here next
    



