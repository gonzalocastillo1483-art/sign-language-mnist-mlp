from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile


CSV_NAMES = {
    "sign_mnist_train.csv": "sign_mnist_train.csv",
    "sign_mnist_test.csv": "sign_mnist_test.csv",
}


def find_member(zip_file: ZipFile, file_name: str) -> str:
    exact_matches = [name for name in zip_file.namelist() if name == file_name]
    if exact_matches:
        return exact_matches[0]

    suffix_matches = [name for name in zip_file.namelist() if name.endswith("/" + file_name)]
    if suffix_matches:
        return suffix_matches[0]

    raise FileNotFoundError(f"Could not find {file_name} in the ZIP file.")


def extract_csvs(zip_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path) as zip_file:
        for source_name, target_name in CSV_NAMES.items():
            member = find_member(zip_file, source_name)
            target_path = output_dir / target_name
            with zip_file.open(member) as source, target_path.open("wb") as target:
                target.write(source.read())
            print(f"Extracted {member} -> {target_path}")


def main():
    parser = ArgumentParser(description="Extract Sign Language MNIST CSV files.")
    parser.add_argument(
        "--zip",
        dest="zip_path",
        default=str(Path.home() / "Downloads" / "Sign Language MNIST.zip"),
        help="Path to Sign Language MNIST.zip",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="data/raw",
        help="Folder where CSV files will be extracted",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    extract_csvs(zip_path, output_dir)


if __name__ == "__main__":
    main()
