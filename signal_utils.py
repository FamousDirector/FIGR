import numpy as np
from scipy import signal
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt
import pycwt


def get_multi_channel_spectral_array(raw_signals, num_of_channels, samp_freq, spectral_type, output_dir, resize=None):
    multi_channel_spectral = []
    min_max_values = []

    # split signal into channels
    channels = np.split(raw_signals, num_of_channels)

    # for each temporal channel
    for i, ch in enumerate(channels):

        min_max_values.append([np.min(ch), np.max(ch)])

        # calculate spectrogram image
        signal_length = len(ch)
        if spectral_type == "stft":
            _, _, spectral = signal.stft(ch, samp_freq,
                                         nperseg=int(signal_length / 1),
                                         noverlap=int(signal_length / 1) - 1
                                         )
        elif spectral_type == "cwt":
            spectral, sj, _, _, _, _ = pycwt.cwt(ch, 1 / samp_freq,
                                                 dj=1/16,
                                                 wavelet=pycwt.Morlet(6)
                                                 )
            np.savetxt('%s/sj_values.csv' % output_dir, sj, delimiter=',')
        else:
            raise Exception('Spectral type %s is not supported' % spectral_type)

        # generate spectrogram image
        min_intensity = np.min(spectral.real)
        max_intensity = np.max(spectral.real)

        img_array = ((spectral.real - min_intensity) / (max_intensity - min_intensity) * 255)

        img = Image.fromarray(img_array).convert('L')
        img_array = np.array(img)
        multi_channel_spectral.append(img_array)

    multi_channel_spectral = np.array(multi_channel_spectral)  # convert to np array
    multi_channel_spectral = multi_channel_spectral.transpose((1, 2, 0))  # convert to HWC for conformity

    np.savetxt('%s/hwc.csv' % output_dir,
               np.array(multi_channel_spectral.shape), delimiter=',')  # save min/max values for signals

    np.savetxt('%s/min_max_values.csv' % output_dir,
               np.array(min_max_values), delimiter=',')  # save min/max values for signals

    return multi_channel_spectral


def reconstruct_signals(min_max_values_path, num_of_channels, samp_freq, num_channel_samples, spectral_type, output_dir):
    """
    This function creates a csv file that contains the reconstructed temporal signals that were generated with SinGAN
    """
    signals = []

    min_max_values = np.loadtxt('%s/min_max_values.csv' % min_max_values_path, delimiter=',')

    for filename in os.listdir(output_dir):
        if filename.endswith(".npz"):

            npzfile = np.load(os.path.join(output_dir, filename))
            key = sorted(npzfile.files)[0]
            spectral_array = npzfile[key].transpose(2, 0, 1)

            s = np.array([])

            for j in range(num_of_channels):
                img_array = np.array(spectral_array[j])

                if spectral_type == "stft":
                    _, xrec = signal.istft(img_array, samp_freq,
                                           nperseg=int(num_channel_samples / 1),
                                           noverlap=int(num_channel_samples / 1) - 1
                                           )

                elif spectral_type == "cwt":
                    sj = np.loadtxt('%s/sj_values.csv' % min_max_values_path, delimiter=',')
                    xrec = pycwt.icwt(img_array, sj, 1 / samp_freq,
                                      dj=1/16,
                                      wavelet=pycwt.Morlet(6)
                                      )
                    xrec = xrec.real
                else:
                    raise Exception('Spectral type %s is not supported' % spectral_type)

                x_norm = ((xrec - np.min(xrec)) / (np.max(xrec) - np.min(xrec)))
                x_renorm = (x_norm * (min_max_values[j][1] - min_max_values[j][0])) + min_max_values[j][0]
                s = np.append(s, x_renorm)

            signals.append(s)

    with open((output_dir + '/signals.csv'), 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')
        for sig in signals:
            wr.writerow(sig)


def plot_signals(num_of_channels, output_dir):
    # read in signals
    signals = np.genfromtxt(output_dir + '/signals.csv', delimiter=',')

    # plot figures
    for i in range(signals.shape[0]):
        channels = np.split(signals[i], num_of_channels)
        fig, ax = plt.subplots(1, num_of_channels)

        for j, col in enumerate(ax):
            x = list(range(len(channels[j])))
            y = channels[j]
            col.plot(x, y)

        plt.savefig(output_dir + '/reconstructed_signals_' + str(i) + '.png', dpi=300)
        plt.close()

