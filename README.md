
# shorts-auto-editor

it looks for audio track 2/3 (mic then desktop) and transcribes them. styles are easy to change in the subtitles editor

# To Do

get a title thing working

## Installation

### Prerequisites

- Python 3.6 or higher
- FFmpeg

[FFMPEG installation tutorials](https://gist.github.com/barbietunnie/47a3de3de3274956617ce092a3bc03a1) 

### Steps

1. Create a virtual environment and activate it (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    - I personally use Conda for environment cause its first thing I figured out with python virtual envs

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Install tkinter
   
   For Mac OS
   ```sh
   brew install python-tk
   ```

   For Linux (Debian based)
   ```
   sudo apt-get install python3-tk
   ```

   For Windows

   Usually comes pre-installed with Python, if not see the [official documentation](https://tkdocs.com/tutorial/install.html)
   
## Usage

Run the main script:
```sh
python main.py
```

