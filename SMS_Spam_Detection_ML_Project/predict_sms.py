from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parent
model=joblib.load(ROOT/'models'/'best_sms_spam_model.joblib')

def predict_sms(message):
 label=model.predict([message])[0]
 # Pipeline's TF-IDF + Naive Bayes; confidence shown as model probability, not certainty
 probs=model.predict_proba([message])[0]
 classes=model.classes_
 confidence=float(probs[list(classes).index(label)])
 return label, confidence

if __name__=='__main__':
 print('SMS Spam Detector. Type a message (or exit to quit).')
 while True:
  msg=input('\nSMS: ').strip()
  if msg.lower() in {'exit','quit'}: break
  if not msg: continue
  label,conf=predict_sms(msg)
  print(f'Prediction: {label.upper()} | model probability: {conf:.1%}')
