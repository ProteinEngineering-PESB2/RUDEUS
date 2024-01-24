from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def random_under_sampling_split(X, y, random_state = None):
    """Get a balanced random sample of data and splits it into training and testing datasets."""
    if random_state is not None:
        rus = RandomUnderSampler(random_state = random_state)
    else:
        rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    return train_test_split(X, y, random_state=1)

def binarize_target(y):
    """Binarize target"""
    encoder = LabelEncoder()
    return encoder.fit_transform(y)
