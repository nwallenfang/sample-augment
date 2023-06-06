import generate as gen
from sample_augment.utils.paths import project_path


def main():
    """
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
    """
    generated_dir = project_path('data/generated', create=True)
    gen.generate_images(out_dir=generated_dir, seeds=[i for i in range(3)], class_idx=1,
                        network_pkl=
                        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl")


if __name__ == '__main__':
    main()
