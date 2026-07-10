# CTI Phishing URL Detector Chrome Extension

Unpacked Chrome extension for CTI Lab Exercise 3. It classifies the active tab URL by calling the existing Flask detector at `http://127.0.0.1:5000/predict`.

## Run the detector

From `CTI LAB/03-phishing-url-detector`:

```powershell
python run.py web
```

Or from the SEM 9 project root:

```powershell
.venv\Scripts\python.exe "CTI LAB\03-phishing-url-detector\run.py" web
```

The extension expects the Flask app to be available on port `5000`.

## Load in Chrome

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select this folder: `CTI LAB/03-phishing-url-detector/chrome-extension`.

## Use

- Open any `http` or `https` page.
- Click the extension icon.
- Use `Classify Current Tab`, or enter a URL manually.
- The toolbar badge shows `OK`, `PHISH`, or `OFF` when the detector is not running.

This extension reports classifications only. It does not block navigation or send browsing data anywhere except the local CTI 03 Flask app.
