import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import pandas as pd

# Place credentials in same folder and change variable
FIREBASE_CREDENTIALS_PATH = "credentials/biome-app-2-firebase-adminsdk-soxoo-b3f1bf7e27.json"
OUTPUT_PATH = "firebase-measurements.csv"

cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)

db = firestore.client()  # this connects to our Firestore database
collection = db.collection("diameter-measurements")  # opens 'diameter-measurements' collection

results = np.empty((0,3), float)
# results = [[1,2,3]]

for document in collection.get():
    entryDict = document.to_dict()
    distance = entryDict['distance']
    diameter = entryDict['diameter']
    # Multiple spellings due to change in app code
    pixelWidth = max(entryDict.get("pixelwidth", 0), entryDict.get("pixelWidth", 0))
    results = np.vstack([results, [distance, pixelWidth, diameter]])

df = pd.DataFrame(results, columns=["distance", "pixel_width", "diameter"])

print(df)

df.to_csv(OUTPUT_PATH, sep=",", header=True, index=False)
