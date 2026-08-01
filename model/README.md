# Vibecode Detector — local web app

A one-page web app that scores screenshots of websites/apps on how likely they are to be
"vibecoded", using the ResNet18 model you trained in the Colab notebook. Runs entirely on
your machine — no image ever leaves your computer.

## 1. Add your model files

From the Colab notebook's final step, you downloaded a zip containing:

```
final_model_state_dict.pt
metadata.json
```

Copy both files into the `model/` folder here, so it looks like:

```
vibecoded_webapp/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── model/
    ├── final_model_state_dict.pt
    └── metadata.json
```

## 2. Install dependencies

```bash
cd vibecoded_webapp
pip install -r requirements.txt
```

(A virtual environment is recommended: `python -m venv venv && source venv/bin/activate` first,
on Windows use `venv\Scripts\activate`.)

## 3. Run it

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## How it works

- Upload one or more screenshots (drag-and-drop or click to browse).
- Click **Analyze**. Each image is preprocessed exactly the way training images were
  (resized + padded to a square, then normalized) and passed through the model.
- Each image gets its own probability of being "vibecoded", shown with a small meter.
- The **average probability across every uploaded image** is shown prominently at the top,
  along with the overall verdict — this uses the same classification threshold your
  notebook tuned for the best F1 score (stored in `metadata.json`), not a plain 50% cutoff.

## Notes

- First run may take a few seconds longer while PyTorch loads the model.
- Runs on CPU fine for single/few-image inference — no GPU required.
- If you retrain the model later, just replace the two files in `model/` and restart `app.py`.
