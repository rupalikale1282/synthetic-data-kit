import os
import shutil
import requests
import lance
from synthetic_data_kit.cli import app
from typer.testing import CliRunner

# URL of a sample PDF with images for testing
PDF_URL = "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf"
OUTPUT_DIR = "example_output"


def test_single_file_mode():
    """Tests the single file processing mode."""
    print("--- Testing Single File Mode ---")
    # Define paths for this test
    pdf_filename = "sample_single.pdf"
    output_dir_single = os.path.join(OUTPUT_DIR, "single_file")
    pdf_path = os.path.join(output_dir_single, pdf_filename)
    lance_path = os.path.join(output_dir_single, "sample_single.lance")
    json_path = os.path.join(output_dir_single, "sample_single.json")

    # Clean and create directory
    if os.path.exists(output_dir_single):
        shutil.rmtree(output_dir_single)
    os.makedirs(output_dir_single)

    # Download the test PDF
    response = requests.get(PDF_URL)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    # Run ingest on a single file
    runner = CliRunner()
    result = runner.invoke(
        app, ["ingest", pdf_path, "--output-dir", output_dir_single, "--multimodal"]
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert os.path.exists(lance_path)
    print("Single file ingestion successful!")

    # Run create on the single Lance file
    result = runner.invoke(
        app,
        ["create", lance_path, "--output-dir", output_dir_single, "--type", "multimodal-qa"],
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert os.path.exists(json_path)
    print("Single file QA pair generation successful!")


def test_folder_mode():
    """Tests the folder processing mode."""
    print("\n--- Testing Folder Mode ---")
    # Define paths for this test
    pdf_folder = os.path.join(OUTPUT_DIR, "pdf_folder")
    lance_dir = os.path.join(OUTPUT_DIR, "lance_files")
    json_dir = os.path.join(OUTPUT_DIR, "json_files")

    # Clean and create directories
    for dir_path in [pdf_folder, lance_dir, json_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # Download the test PDF multiple times
    num_files = 3
    for i in range(num_files):
        response = requests.get(PDF_URL)
        pdf_path = os.path.join(pdf_folder, f"sample_{i}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(response.content)

    # Run ingest on the folder
    runner = CliRunner()
    result = runner.invoke(
        app, ["ingest", pdf_folder, "--output-dir", lance_dir, "--multimodal"]
    )
    print(result.stdout)
    assert result.exit_code == 0
    lance_files = [f for f in os.listdir(lance_dir) if f.endswith(".lance")]
    assert len(lance_files) == num_files
    print("Folder ingestion successful!")

    # Run create on the directory of Lance files
    result = runner.invoke(
        app, ["create", lance_dir, "--output-dir", json_dir, "--type", "multimodal-qa"]
    )
    print(result.stdout)
    assert result.exit_code == 0
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    assert len(json_files) == num_files
    print("Folder QA pair generation successful!")


def main():
    """Run both single file and folder mode tests."""
    # Set the API key
    os.environ["API_ENDPOINT_KEY"] = "LLM|704426635437672|3nFowHkWPXPZWYaepVCJC0Z3GMw"

    # Clean and create the main output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    test_single_file_mode()
    test_folder_mode()


if __name__ == "__main__":
    main()