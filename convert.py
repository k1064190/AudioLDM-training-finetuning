from functools import partial

import librosa
import os
from tqdm import tqdm
import pretty_midi
import fluidsynth
import numpy as np
import soundfile as sf
import random
import argparse
from multiprocessing import Pool

parsearg = argparse.ArgumentParser()
parsearg.add_argument('-dur', '--duration', type=int, default=10)
parsearg.add_argument('-proc', '--processes', type=int, default=8)

args = parsearg.parse_args()
AUDIO_DURATION = args.duration * 44100
PROCESSES = args.processes

def delete_mute(sample):
    crit = 0.01 * max(np.abs(sample))
    mute_trimmed_sample = sample[np.where(np.abs(sample) > crit)]
    return mute_trimmed_sample


def trim_audio(sample):
    non_silent_indices = librosa.effects.split(sample, top_db=20, frame_length=AUDIO_DURATION,
                                               hop_length=AUDIO_DURATION)

    trimmed_sample = np.array([])

    for indices in non_silent_indices:
        ns = sample[indices[0]:indices[1]]
        trimmed_sample = np.concatenate([trimmed_sample, ns])

    return trimmed_sample


def select_frame(sample, criterion='random'):
    audio_frames = librosa.util.frame(sample, frame_length=AUDIO_DURATION, hop_length=AUDIO_DURATION // 4, axis=0)

    if criterion == 'random':
        random_idx = random.randint(0, audio_frames.shape[0] - 1)

        return audio_frames[random_idx]

    elif criterion == 'max':
        max_ste = 0
        max_idx = -1
        for i, frame in enumerate(audio_frames):
            ste = np.sum(np.square(frame))

            if ste > max_ste:
                max_ste = ste
                max_idx = i

        return audio_frames[max_idx]

    else:
        raise ValueError("Criterion must be random or max")


def list_existing_pathes(root_dir):
    midi_pathes = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, str(folder_name))
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mid'):
                file_path = os.path.join(folder_path, file_name)
                midi_pathes.append(file_path)

    return midi_pathes

def convert(midi_file_paths, out_path):
    print("Converting ", len(midi_file_paths), " files")
    for midi_file_path in midi_file_paths:
        midi_file_name = midi_file_path.split("/")[-1]
        midi_id = os.path.splitext(midi_file_name)[0]
        out_file = os.path.join(out_path, f'{midi_id}.wav')

        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        audio_data = midi_data.fluidsynth()

        audio_data = trim_audio(audio_data)

        if len(audio_data) < AUDIO_DURATION:
            continue

        audio_data = select_frame(audio_data, criterion='random')

        sf.write(out_file, audio_data, 44100, subtype='PCM_16', format='WAV')

def main():
    # midi_path = 'extract'
    root_dir = 'extracted_instruments'
    out_path = 'wavs_10s_all'

    os.makedirs(out_path, exist_ok=True)

    midi_pathes = list_existing_pathes(root_dir)
    midi_pathes = np.array(midi_pathes)
    print("수집된 MIDI 파일의 수:", len(midi_pathes))
    splited_data = np.array_split(midi_pathes, PROCESSES)

    # pop out first element and make it pbar
    for i in range(PROCESSES):
        print(f"Process {i} has {len(splited_data[i])} MIDI files")
    new_midi_pathes = splited_data[1:]
    main_pathes = splited_data[0]
    midi_pbar = tqdm(main_pathes, desc='Converting MIDI to WAV', miniters=100, total=len(main_pathes))

    run = partial(convert, out_path=out_path)

    # print("수집된 MIDI 파일의 수:", len(new_midi_pathes))
    # print("첫 번째 MIDI 파일 경로:", new_midi_pathes[0])
    # convert(new_midi_pathes, out_path)
    # parmap.map(run, splited_data, pm_pbar=True, pm_processes=PROCESSES)
    p = Pool(PROCESSES)
    p.apply_async(run, args=(midi_pbar,))
    for midi_path in new_midi_pathes:
        p.apply_async(run, args=(midi_path,))

    p.close()
    p.join()


if __name__ == '__main__':
    main()
