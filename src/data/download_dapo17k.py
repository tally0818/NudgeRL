from datasets import load_dataset
from pathlib import Path


def download_and_save(output_path: Path, subset: str = "en") -> Path:
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", subset)
    output_path.mkdir(parents=True, exist_ok=True)
    save_dir = output_path / "dapo-17k"
    print(f"Saving DAPO-Math-17k to: {save_dir}")
    ds.save_to_disk(str(save_dir))
    print("Save complete.")
    return save_dir


def main():
    download_and_save(Path("data"), subset="en")


if __name__ == "__main__":
    main()
