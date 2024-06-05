import shutil
import os

import argparse
import yaml
import torch

from utilities.data.dataset import AudioDataset

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from audioldm_train.utilities.tools import get_restore_step, build_dataset_json_from_list
from audioldm_train.utilities.model_util import instantiate_from_config
from audioldm_train.utilities.tools import build_dataset_json_from_list
from audioldm_train.utilities.audio import TacotronSTFT
from audioldm_train.utilities.audio.tools import wav_to_fbank

import contextlib
import wave

def get_duration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def round_up_duration(duration):
    return int(round(duration / 2.5) + 0.25) * 2.5


def infer(data, configs, exp_name, ts=0.5, style=None):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    val_dataset = AudioDataset(
        configs, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json
    )

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    val_dataset = AudioDataset(
        configs, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
    )

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(output_dir, 'results', 'musicclap')

    guidance_scale = configs["model"]["params"]["evaluation_params"][
        "unconditional_guidance_scale"
    ]
    ddim_sampling_steps = configs["model"]["params"]["evaluation_params"][
        "ddim_sampling_steps"
    ]
    n_candidates_per_samples = configs["model"]["params"]["evaluation_params"][
        "n_candidates_per_samples"
    ]

    resume_from_checkpoint = configs["reload_from_ckpt"]

    checkpoint = torch.load(resume_from_checkpoint)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()

    if style is not None:
        fn_STFT = TacotronSTFT(
            configs["preprocessing"]["stft"]["filter_length"],
            configs["preprocessing"]["stft"]["hop_length"],
            configs["preprocessing"]["stft"]["win_length"],
            configs["preprocessing"]["mel"]["n_mel_channels"],
            configs["preprocessing"]["audio"]["sampling_rate"],
            configs["preprocessing"]["mel"]["mel_fmin"],
            configs["preprocessing"]["mel"]["mel_fmax"],
        )
        mels = []
        for e in style:
            original_audio_file_path = e
            audio_file_duration = get_duration(original_audio_file_path)
            duration = round_up_duration(audio_file_duration)
            mel, log_mel, waveform = wav_to_fbank(original_audio_file_path, target_length=int(duration * 102.4),
                                              fn_STFT=fn_STFT)
            mels.append(mel)
        mel = torch.stack(mels).cuda()
    else:
        mel = None

    print(f"data: {data}")
    print(f"ts: {ts}")
    print(f"guidance_scale: {guidance_scale}")
    print(f"ddim_sampling_steps: {ddim_sampling_steps}")
    print(f"n_candidates_per_samples: {n_candidates_per_samples}")
    latent_diffusion.generate_sample(
        val_loader,
        unconditional_guidance_scale=guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_gen=n_candidates_per_samples,
        ts=ts,
        x_T=mel,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        nargs="+",
        help="The filelist that contain captions (and optionally filenames)",
    )

    parser.add_argument(
        "-s",
        "--style",
        type=str,
        required=False,
        nargs="+",
        help="The filelist for style transfer",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="output",
        help="The output directory",
    )
    parser.add_argument(
        "-ckpt",
        "--reload_from_ckpt",
        type=str,
        required=True,
        help="the checkpoint path for the model",
    )
    parser.add_argument(
        "--transfer_strength",
        "-ts",
        type=float,
        required=False,
        default=0.5,
        help="The transfer strength",
    )

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    assert len(args.style) == len(args.prompt), "The number of styles and prompts must be the same"

    config_yaml = args.config_yaml
    # {
    #     "data": [
    #         {
    #             "wav": "path/to/wav",
    #             "caption": "caption",
    #         }
    #     ]
    # }
    data = []
    for idx, prompt in enumerate(args.prompt):
        data.append({"wav": "", "caption": prompt})
    dataset_json = {
        "data": data
    }
    output_dir = args.output_dir
    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt != None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    infer(dataset_json, config_yaml, output_dir, args.transfer_strength, args.style)
