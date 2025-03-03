import argparse, os
import numpy as np
import glob
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default='init_latent',
        help="Target variable",
    )
    parser.add_argument(
        "--roi",
        default=['VC'],
        type=str,
        nargs="*",
        help="use roi name",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default='sub-3',
    )

    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi

    backend = set_backend("numpy", on_error="warn")
    subject=opt.subject

    if target == 'c' or target == 'init_latent':
        alpha = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]
    else: 
        alpha = [10000, 20000, 40000]

    ridge = RidgeCV(alphas=alpha)

    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )    
    mridir = f'../../mindart/semfeat/DIR/'
    featdir = f'../../mindart/dirfeat/{target}/'
    savedir = f'../../mindart/decoded/DIR/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    cX = np.load(f'{mridir}/{subject}/train_fmri_features.npy', allow_pickle = True).item()
    cX_te = np.load(f'{mridir}/{subject}/test_fmri_features.npy', allow_pickle = True).item()
    
    Y_dir = np.array(glob.glob(f'{featdir}/train/*.npy'))
    Y_ids = [train_path.split('train\\')[-1].split('.')[0] for train_path in Y_dir]
    Y = np.vstack([np.load(train_path).astype("float32") for train_path in Y_dir])
    X = np.vstack([np.mean(np.vstack(cX[y_ids]).reshape(len(cX[y_ids]), -1), 0) for y_ids in Y_ids])

    Y_te_dir = np.array(glob.glob(f'{featdir}/test/*.npy'))
    Y_te_ids = [test_path.split('test\\')[-1].split('.')[0] for test_path in Y_te_dir]
    Y_te = np.vstack([np.load(test_path).astype("float32") for test_path in Y_te_dir])
    X_te = np.vstack([np.vstack(cX_te[y_te_ids]).flatten() for y_te_ids in Y_te_ids])

    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    pipeline.fit(X, Y)
    scores = pipeline.predict(X_te)
    prediction = {}
    for i in range(scores.shape[0]):
        prediction[Y_te_ids[i]] = scores[i]
    np.save(f'{savedir}/prediction.npy', prediction)

if __name__ == "__main__":
    main()
