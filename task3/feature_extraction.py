import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.decomposition import PCA, KernelPCA
from biosppy.signals import ecg  # preprocessing ECG signal
import math


# test with PCA + kernel PCA feature
class PCAFeature(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.95, kernel='rbf', gamma=0.5):
        self.pca = PCA(n_components=n_components)
        self.kpca = KernelPCA(kernel=kernel, gamma=gamma)

    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        X_kpca = self.kpca.fit_transform(X)
        X_new = np.hstack((X_pca, X_kpca))
        print("PCA Feature dimension:")
        print(X_new.shape)
        return X_new


class BioFeature(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate=300,
                 threshold=0.48, segstart=90, segend=110):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.segstart = segstart  # Q-R interval
        self.segend = segend      # R-S interval

    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        T = 1.0 / self.sample_rate
        bio_feature = np.empty(1)
        for i in range(X.shape[0]):
            Xi = X[i, :]
            rpeaks = ecg.engzee_segmenter(Xi,
                                          sampling_rate=300,
                                          threshold=self.threshold)
            rpeaks = rpeaks[0]
            # 1. compute the mean of DFT for one QRS wave
            QR_int = self.segstart    # Q-R interval
            RS_int = self.segend      # R-S interval
            m_idx = X.shape[1] - RS_int
            idx = np.where((rpeaks > QR_int) & (rpeaks < m_idx))
            idx = idx[0]
            new_rpeaks = rpeaks[idx]
            _fft = np.zeros(QR_int + RS_int)
            for j in range(len(new_rpeaks)):
                # extract segment around rpeaks
                QRS = Xi[new_rpeaks[j] - QR_int:new_rpeaks[j] + RS_int]
                # compute the mean of DFT
                fft = np.fft.fft(QRS)
                fft_amp = np.absolute(fft)
                if j == 0:
                    _fft = fft_amp
                else:
                    _fft = np.vstack((_fft, fft_amp))
            fft_mean = np.mean(_fft, axis=0)
            if len(new_rpeaks) == 0 or len(new_rpeaks) == 1:
                fft_mean = _fft

            # 2. compute the statistics of heart beat
            out = ecg.ecg(signal=Xi, sampling_rate=300,
                          show=False)
            [ts, sig, rpeaks, temp_ts, temp, hr_ts, heart_rate] = out
            heart_stats = [0, 0, 0, 0]
            if len(heart_rate) != 0:
                heart_stats[0] = np.mean(heart_rate)
                heart_stats[1] = np.var(heart_rate)
                heart_stats[2] = np.amin(heart_rate)
                heart_stats[3] = np.amax(heart_rate)
            heart_stats = np.asarray(heart_stats)

            # 3. compute the statistics of R-R interval
            RR_int = []
            for k in range(1, len(rpeaks)):
                RR_int.append(T * (rpeaks[k] - rpeaks[k-1]))
            RR_int = np.asarray(RR_int)
            RR_stats = [0, 0, 0, 0]
            if len(RR_int) != 0:
                RR_stats[0] = np.mean(RR_int)
                RR_stats[1] = np.var(RR_int)
                RR_stats[2] = np.amin(RR_int)
                RR_stats[3] = np.amax(RR_int)
            RR_stats = np.asarray(RR_stats)

            feature = np.hstack((fft_mean, heart_stats))
            feature = np.hstack((feature, RR_stats))
            if i == 0:
                bio_feature = feature
            else:
                bio_feature = np.vstack((bio_feature, feature))
        return bio_feature


class FFTFeature(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate=300, save_path=None):
        self.sample_rate = sample_rate
        self.save_path = save_path

    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        fft_out = np.fft.fft(X)
        X_new = np.absolute(fft_out)
        print("FFT feature space:")
        print(X_new.shape)
        return X_new


class MixFeature(BaseEstimator, TransformerMixin):
    ''' PCA + BioFeature '''
    def __init__(self, sample_rate=300, threshold=0.48, n_comp=0.95):
        self.pca = PCA(n_components=n_comp)
        self.sample_rate = sample_rate
        self.threshold = threshold

    def fit(self, X, y):
        pass


class HRVFeature(BaseEstimator, TransformerMixin):
    """ Adpated from <<van Gent, P. (2016).
    Analyzing a Discrete Heart Rate Signal Using Python.
    A tech blog about fun things with Python and embedded electronics>>"""
    """ feature includes: BPM, IBI, SDNN. SDSD, RMSSD, pNN50, pNN20 """
    def __init__(self, sample_rate=300,
                 start=40, end=50, save_path=None):
        self.sample_rate = sample_rate
        self.segstart = start
        self.segend = end
        self.save_path = save_path

    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        T = 1.0 / self.sample_rate
        bio_feature = np.empty(1)
        for i in range(X.shape[0]):
            Xi = X[i, :]
            # extract the R-peaks positions
            out = ecg.ecg(signal=Xi, sampling_rate=300,
                          show=False)
            [ts, sig, rpeaks, temp_ts, temp, hr_ts, heart_rate] = out
            # 1. Extract Time Domain Measures
            # Compute the R-R interval (ms)
            RR_int = []
            for k in range(1, len(rpeaks)):
                RR_int.append(1000.0 * T * (rpeaks[k] - rpeaks[k - 1]))
            RR_int = np.asarray(RR_int)
            # Compute the R-R interval difference
            RR_diff = []    # difference between adajacent RR interval
            RR_sqdiff = []  # squared difference of RR interval
            for k in range(1, len(RR_int)):
                m_diff = RR_int[k] - RR_int[k - 1]
                RR_diff.append(abs(m_diff))
                RR_sqdiff.append(math.pow(m_diff, 2))
            RR_diff = np.asarray(RR_diff)
            RR_sqdiff = np.asarray(RR_sqdiff)

            bio_feature = np.zeros(10)
            if len(RR_int) >= 1:
                # 1.ibi measure: the mean of the R-R interval
                ibi = np.mean(RR_int)
                bio_feature[0] = ibi

                # 2.sdn measure: the standard deviation of R-R intervals
                sdnn = np.std(RR_int)
                bio_feature[1] = sdnn

                # 3. max of the R-R interval
                bio_feature[2] = np.amax(RR_int)

                # data = pd.Series(RR_int)
                # arma_mod4 = sm.tsa.ARMA(data, (4,0)).fit()
                # bio_feature[3] = arma_mod4.params[0]
                # bio_feature[4] = arma_mod4.params[1]
                # bio_feature[5] = arma_mod4.params[2]
                # bio_feature[6] = arma_mod4.params[3]

                if len(RR_diff) >= 1:
                    # 5.sdsd measure: the standard deviation of the R-R diff
                    sdsd = np.std(RR_diff)
                    bio_feature[3] = sdsd

                    # 6. the mean , min, max of the R-R difference
                    bio_feature[4] = np.mean(RR_diff)
                    bio_feature[5] = np.amax(RR_diff)

                    # 4.rsmd measure: the root mean square of R-R differences
                    rmssd = np.sqrt(np.mean(RR_sqdiff))
                    bio_feature[6] = rmssd

                    # 5. pnn20/50: the percentage of differences > 50/20 ms
                    nn20 = [x for x in RR_diff if (x > 20)]
                    nn50 = [x for x in RR_diff if (x > 50)]
                    pnn20 = float(len(nn20)) / float(len(RR_diff))
                    pnn50 = float(len(nn50)) / float(len(RR_diff))
                    bio_feature[7] = pnn20
                    bio_feature[8] = pnn50

                # 6. bpm measure: the average heart rate per minute
                # bpm = np.mean(heart_rate)
                bpm = 60000 / np.mean(RR_int)
                bio_feature[9] = bpm

                # 7. extract the QRS waves and use the mean as feature
                QR_int = self.segstart    # Q-R interval
                RS_int = self.segend      # R-S interval
                m_idx = X.shape[1] - RS_int
                idx = np.where((rpeaks > QR_int) & (rpeaks < m_idx))
                idx = idx[0]
                new_rpeaks = rpeaks[idx]
                nQRS = np.zeros(QR_int + RS_int)
                for j in range(len(new_rpeaks)):
                    # extract segment around rpeaks
                    QRS = Xi[new_rpeaks[j] - QR_int:new_rpeaks[j] + RS_int]
                    # QRS = np.absolute(np.fft.fft(QRS))
                    # compute the mean of DFT
                    if j == 0:
                        nQRS = QRS
                    else:
                        nQRS = np.vstack((nQRS, QRS))
                QRS_mean = np.mean(nQRS, axis=0)
                if len(new_rpeaks) == 0 or len(new_rpeaks) == 1:
                    QRS_mean = nQRS
                bio_feature = np.hstack((bio_feature, QRS_mean))

            for j in range(len(bio_feature)):
                if np.isnan(bio_feature[j]):
                    print("Error in extracting sample")
                    print(i)
                    print(":found NaN in bio_feature")
            if i == 0:
                features = bio_feature
            else:
                features = np.vstack((features, bio_feature))
        return features
