import os
import torch



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