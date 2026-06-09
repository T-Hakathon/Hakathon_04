
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

TRAIN, TEST = "dataset_C_training.csv", "dataset_C_testing.csv"
ID, TARGET = "respondent_id", "covid_vaccine"
GROUP = "T"        
ORDER = 4

train = pd.read_csv(TRAIN); test = pd.read_csv(TEST)
y = train[TARGET].astype(int)
CAT = ["age_group","education","race","sex","income_poverty","marital_status",
       "rent_or_own","employment_status","census_msa","employment_sector"]

def engineer(df):
    df = df.copy()
    beh = ["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask",
           "behavioral_wash_hands","behavioral_large_gatherings","behavioral_outside_home",
           "behavioral_touch_face"]
    df["behavior_count"] = df[beh].sum(axis=1, skipna=True)
    df["opinion_net"] = df["opinion_covid_vacc_effective"].fillna(3) - df["opinion_covid_sick_from_vacc"].fillna(3)
    df["risk_x_eff"] = df["opinion_covid_risk"].fillna(3) * df["opinion_covid_vacc_effective"].fillna(3)
    df["household_total"] = df["household_adults"].fillna(0) + df["household_children"].fillna(0)
    df["missing_count"] = df.isna().sum(axis=1)
    return df

tr, ts = engineer(train), engineer(test)
FEATS = [c for c in tr.columns if c not in (ID, TARGET)]

def numeric_matrix():
    X = tr[FEATS].copy(); Xt = ts[FEATS].copy()
    for c in CAT:
        cats = X[c].astype("category")
        m = {v: i for i, v in enumerate(cats.cat.categories)}
        X[c] = X[c].map(m); Xt[c] = Xt[c].map(m)
    return X.astype(float), Xt.astype(float)

def best_threshold(yt, p):
    ths = np.linspace(0.10, 0.70, 61)
    f1s = [f1_score(yt, (p > t).astype(int)) for t in ths]
    i = int(np.argmax(f1s)); return ths[i], f1s[i]

skf = StratifiedKFold(5, shuffle=True, random_state=42)

Xn, Xtn = numeric_matrix()
oof = np.zeros(len(Xn)); pred = np.zeros(len(Xtn))
imp = SimpleImputer(strategy="median")
for tri, vai in skf.split(Xn, y):
    Xi = imp.fit_transform(Xn.iloc[tri]); Xvi = imp.transform(Xn.iloc[vai]); Xti = imp.transform(Xtn)
    m = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=42)
    m.fit(Xi, y.iloc[tri])
    oof[vai] = m.predict_proba(Xvi)[:, 1]
    pred += m.predict_proba(Xti)[:, 1] / 5

thr, f1 = best_threshold(y, oof)
labels = (pred > thr).astype(int)
sub = pd.DataFrame({ID: test[ID], TARGET: labels})
fname = f"challenge_submission_group_{GROUP}_order_{ORDER}.csv"
sub.to_csv(fname, index=False)
print(f"Extra Trees  |  CV F1 = {f1:.4f}  |  threshold = {thr:.2f}")
print(f"Wrote {fname}  ({int(labels.sum())} predicted vaccinated of {len(labels)})")
