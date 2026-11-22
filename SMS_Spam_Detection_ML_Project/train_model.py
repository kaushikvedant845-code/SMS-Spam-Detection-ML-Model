from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'spam.csv'
MODELS=ROOT/'models'; RESULTS=ROOT/'results'
MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
# Read source CSV, discard irrelevant extra columns, and clean rows
raw=pd.read_csv(DATA, encoding='latin-1')
df=raw[['v1','v2']].rename(columns={'v1':'label','v2':'message'}).dropna()
df['label']=df['label'].astype(str).str.strip().str.lower()
df['message']=df['message'].astype(str).str.strip()
df=df[df.label.isin(['ham','spam']) & df.message.ne('')].drop_duplicates().reset_index(drop=True)
X_train,X_test,y_train,y_test=train_test_split(df.message,df.label,test_size=0.20,random_state=42,stratify=df.label)
models={
 'Multinomial Naive Bayes':Pipeline([('tfidf',TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),min_df=1)),('classifier',MultinomialNB())]),
 'Logistic Regression':Pipeline([('tfidf',TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),min_df=1)),('classifier',LogisticRegression(max_iter=2000,class_weight=None))])
}
summary={ 'dataset':{'total_rows_after_cleaning':int(len(df)),'ham':int((df.label=='ham').sum()),'spam':int((df.label=='spam').sum()),'train_rows':int(len(X_train)),'test_rows':int(len(X_test))},'models':{} }
best_name=None; best_f1=-1
for name,pipe in models.items():
 pipe.fit(X_train,y_train)
 pred=pipe.predict(X_test)
 metrics={'accuracy':float(accuracy_score(y_test,pred)),'spam_precision':float(precision_score(y_test,pred,pos_label='spam',zero_division=0)),'spam_recall':float(recall_score(y_test,pred,pos_label='spam',zero_division=0)),'spam_f1':float(f1_score(y_test,pred,pos_label='spam',zero_division=0)),'confusion_matrix_labels':['ham','spam'],'confusion_matrix':confusion_matrix(y_test,pred,labels=['ham','spam']).tolist(),'classification_report':classification_report(y_test,pred,labels=['ham','spam'],output_dict=True,zero_division=0)}
 summary['models'][name]=metrics
 if metrics['spam_f1']>best_f1: best_f1=metrics['spam_f1']; best_name=name; best_pipe=pipe
 joblib.dump(pipe,MODELS/(name.lower().replace(' ','_')+'.joblib'))
summary['best_model']=best_name
joblib.dump(best_pipe,MODELS/'best_sms_spam_model.joblib')
summary['model_file']='models/best_sms_spam_model.joblib'
(RESULTS/'metrics.json').write_text(json.dumps(summary,indent=2))
# Write readable summary
with open(RESULTS/'evaluation_summary.txt','w') as f:
 f.write('SMS SPAM DETECTION - MODEL EVALUATION\n'+'='*45+'\n')
 f.write(f"Rows: {len(df)} | Ham: {(df.label=='ham').sum()} | Spam: {(df.label=='spam').sum()}\n")
 f.write(f"Train: {len(X_train)} | Test: {len(X_test)} (20% test split, random_state=42, stratified)\n\n")
 for name,m in summary['models'].items():
  f.write(name+'\n')
  for key in ['accuracy','spam_precision','spam_recall','spam_f1']:
   f.write(f'  {key}: {m[key]:.4f}\n')
  f.write(f"  confusion matrix (rows actual ham/spam, columns predicted ham/spam): {m['confusion_matrix']}\n\n")
 f.write(f'Best model selected by Spam F1: {best_name}\n')
print(json.dumps(summary['dataset'],indent=2))
for n,m in summary['models'].items(): print(n, {k:round(m[k],4) for k in ['accuracy','spam_precision','spam_recall','spam_f1']},m['confusion_matrix'])
print('BEST:',best_name)
