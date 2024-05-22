import os
import argparse
import json

# Structure of audio files
# audio_path
# ├── 000000.wav
# ├── 000001.wav
# ...

# Using this audio files, we will create metadata for training
# metadata
# ├── dataset_root.json
# ├── datafiles
# │   ├── train_metadata.json
# │   ├── test_metadata.json
# ...

# dataset_root.json
# {
#    "dataset_name": "audio_path",
#    "metadata": {
#        "path": {
#            "dataset_name": {
#                "train": path to train metadata,
#                "test": path to test metadata,
#                "val": path to val metadata,
#                "class_label_indices": we don't need this if we don't have class labels
#            }
#        }
#    }
# }


# train_metadata.json
# {
#    "data" : [
#        {
#            "wav": relative path to audio file (e.g. 000000.wav),
#            "seg_label": path to npy file(not required maybe),
#            "labels": label,
#            "caption": caption,
#        },
#        ...
#    ]
# }
#
# Implement the code that uses text caption if exists or use only waveform data

parsearg = argparse.ArgumentParser(description='Audio to dataset')
parsearg.add_argument('--audio_path', type=str, required=True, help='Path to audio files')
parsearg.add_argument('--metadata_path', type=str, required=False, help='Path to metadata files if not specified, it will be audio path')
parsearg.add_argument('--dataset_name', type=str, required=False, help='Name of dataset if not specified, it will be audio path name')

args = parsearg.parse_args()

audio_path = args.audio_path
metadata_path = args.metadata_path

if metadata_path is None:
    metadata_path = audio_path + '/metadata'
os.makedirs(metadata_path, exist_ok=True)

dataset_name = args.dataset_name
if dataset_name is None:
    dataset_name = os.path.basename(audio_path)
os.makedirs(dataset_name, exist_ok=True)

dataset_root = {
    "dataset_name": dataset_name,
    "metadata": {
        "path": {
            dataset_name: {
                "train": metadata_path + '/train_metadata.json',
                # "test": metadata_path + '/test_metadata.json',
                "val": metadata_path + '/val_metadata.json'
            }
        }
    }
}

audio_list = os.listdir(audio_path)

# 9:1 train, val split
train_list = audio_list[:int(len(audio_list) * 0.9)]
val_list = audio_list[int(len(audio_list) * 0.9):]

train_metadata = {
    "data": []
}

val_metadata = {
    "data": []
}

for i, audio in enumerate(train_list):
    train_metadata["data"].append({
        "wav": audio,
    })  # we don't have labels and captions

for i, audio in enumerate(val_list):
    val_metadata["data"].append({
        "wav": audio,
    })  # we don't have labels and captions


with open(dataset_name + '/dataset_root.json', 'w') as f:
    json.dump(dataset_root, f, indent=4)

with open(metadata_path + '/train_metadata.json', 'w') as f:
    json.dump(train_metadata, f, indent=4)

with open(metadata_path + '/val_metadata.json', 'w') as f:
    json.dump(val_metadata, f, indent=4)

print('Metadata created successfully')
