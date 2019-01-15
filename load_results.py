# Load stuff
import pickle

### Load FINAL error:
with open('networks/networks_dropout/results/final_error_dropout.pkl') as f:  # Python 3: open(..., 'rb')
    final_error_dropout, final_std_dropout = pickle.load(f)

with open('networks/networks_NOdropout/results/final_error_NOdropout.pkl') as f:  # Python 3: open(..., 'rb')
    final_error_NOdropout, final_std_NOdropout = pickle.load(f)


### Load DETERMINISTIC error:
with open('networks/networks_dropout/results/det_validation_error_dropout.pkl') as f:  # Python 3: open(..., 'wb')
    det_error_dropout, det_std_dropout = pickle.load(f)

with open('networks/networks_NOdropout/results/deterministic_error_validationNOdropout.pkl') as f:  # Python 3: open(..., 'wb')
    det_error_NOdropout, det_std_NOdropout = pickle.load(f)


### Load STOCHASTIC error:
with open('networks/networks_dropout/results/stoch_validation_error_dropout.pkl') as f:  # Python 3: open(..., 'wb')
    stoch_error_dropout, stoch_std_dropout = pickle.load(f)

with open('networks/networks_NOdropout/results/stoch_validation_error_NOdropout.pkl') as f:  # Python 3: open(..., 'wb')
    stoch_error_NOdropout, stoch_std_NOdropout = pickle.load(f)
