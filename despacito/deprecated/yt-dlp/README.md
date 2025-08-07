# YouTube Comment Extraction

Deprecated extraction pipeline using yt-dlp.

## Process

### 1. Comment Extraction

Requirements: python, ffmpeg, yt-dlp

Modify URL in `1_grabber.sh` and execute. Downloads all video metadata including comments in JSON format. Process duration varies by comment volume.

### 2. JSON to CSV Conversion

Execute: `python 2_jsontocsv.py seeyouagain.info.json > converted_seeyouagain.csv`

Windows PowerShell requires output redirection modification.

## Status

DEPRECATED. Use modern comment extraction methods.

