import subprocess
import zipfile
import os


def download_file(file_id, output_file, confirm_token='xxx'):
    """
    Downloads a file from Google Drive using curl.

    Parameters:
        file_id (str): The Google file ID of the file to download.
        output_file (str): The name of the output file.
        confirm_token (str): The confirmation token (default is 'xxx').

    Returns:
        None
    """
    # Construct the URL
    url = (
        f'https://drive.usercontent.google.com/download?id={file_id}&'
        f'confirm={confirm_token}'
    )

    # Construct the curl command
    curl_command = [
        'curl',
        url,
        '-o', output_file
    ]

    # Execute the curl command
    result = subprocess.run(curl_command, capture_output=False, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        print(f"Download successful, saved to {output_file}")
    else:
        print("Download failed")


def unzip(zip_file_path):
    """Extracts the contents of a zip file to a directory with the
       same name as the zip file.

    Args:
        zip_file_path (str): The path to the zip file to be extracted.

    Raises:
        FileNotFoundError: If the specified zip file does not exist.
        zipfile.BadZipFile: If the specified file is not a valid zip file.
    """

    with zipfile.ZipFile(zip_file_path, 'r') as z:
        z.extractall(os.path.splitext(zip_file_path)[0])
