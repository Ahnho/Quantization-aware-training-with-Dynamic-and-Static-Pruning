import numpy as np
from scipy.signal import get_window

import torch
import torch.nn as nn
import torch.nn.functional as F

import librosa.util as librosa_util
# from librosa.util import pad_center
from librosa.util import tiny

def pad_center(array, target_length):
    if len(array) >= target_length:
        return array

    pad_width = target_length - len(array)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    padded_array = np.pad(array, (pad_left, pad_right), mode='constant')
    return padded_array


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x



class STFT(torch.nn.Module):
    def __init__(self, filter_length = 1024, hop_length = 512, win_length = None, window = 'hann'):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).
        
        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {1024})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {512})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are bartlett, hann, hamming, blackman, blackmanharris) 
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.filter_length     = filter_length
        self.hop_length        = hop_length
        self.win_length        = win_length if win_length else filter_length
        self.window            = window
        self.forward_transform = None
        self.pad_amount        = int(self.filter_length / 2)
        scale                  = self.filter_length / self.hop_length
        fourier_basis          = np.fft.fft(np.eye(self.filter_length))

        cutoff                 = int((self.filter_length / 2 + 1))
        fourier_basis          = np.vstack([
                                    np.real(fourier_basis[:cutoff, :]),
                                    np.imag(fourier_basis[:cutoff, :])])
        forward_basis          = torch.FloatTensor(fourier_basis[:, None, :])

        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window             = get_window(window, self.win_length, fftbins = True)
        fft_window             = pad_center(fft_window, filter_length)
        fft_window             = torch.from_numpy(fft_window).float()

        forward_basis          *= fft_window # window the bases
        self.register_buffer('forward_basis', forward_basis.float())


    def transform(self, input_data):
        """
        Take input data (audio) to STFT domain.
        
        Arguments:
            input_data {tensor}
                Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            magnitude {tensor}
                Magnitude of STFT with shape (num_batch, 
                num_frequencies, num_frames)
            phase {tensor}
                Phase of STFT with shape (num_batch, 
                num_frequencies, num_frames)
        """
        num_batches      = input_data.shape[0]
        num_samples      = input_data.shape[-1]
        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1), (self.pad_amount, self.pad_amount, 0, 0), mode = 'reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(input_data, self.forward_basis, stride = self.hop_length, padding = 0)

        cutoff    = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        # magnitude = torch.sqrt(real_part**2 + imag_part**2)
        # phase     = torch.atan2(imag_part.data, real_part.data)
        return real_part, imag_part


    def forward(self, input_data):
        """
        Take input data (audio) to STFT domain and then back to audio.
        Arguments:
            input_data {tensor}
                Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            reconstruction {tensor}
                Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        # self.magnitude, self.phase = self.transform(input_data)
        real_part, imag_part = self.transform(input_data)
        real_part = torch.unsqueeze(real_part, dim = 1)
        imag_part = torch.unsqueeze(imag_part, dim = 1)
        return real_part, imag_part



class InverseSTFT(nn.Module):
    def __init__(self, filter_length = 1024, hop_length = 512, win_length = None, window = 'hann'):
        super(InverseSTFT, self).__init__()
        self.filter_length     = filter_length
        self.hop_length        = hop_length
        self.win_length        = win_length if win_length else filter_length
        self.window            = window
        self.forward_transform = None
        self.pad_amount        = int(self.filter_length / 2)
        
        scale                  = self.filter_length / self.hop_length
        fourier_basis          = np.fft.fft(np.eye(self.filter_length))

        cutoff                 = int((self.filter_length / 2 + 1))
        fourier_basis          = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])
        inverse_basis          = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        assert(filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window             = get_window(window, self.win_length, fftbins=True)
        fft_window             = pad_center(fft_window, filter_length)
        fft_window             = torch.from_numpy(fft_window).float()

        inverse_basis          *= fft_window # inverse basis
        self.register_buffer('inverse_basis', inverse_basis.float())


    def inverse(self, real_part, imag_part):
        """
        Call the inverse STFT (iSTFT), given magnitude and phase tensors (or real and imag tensors)
        
        Arguments:
            magnitude {tensor}
                Magnitude of STFT with shape (num_batch, num_frequencies, num_frames)
            phase {tensor}
                Phase of STFT with shape (num_batch, num_frequencies, num_frames)
        
        Returns:
            inverse_transform {tensor}
                Reconstructed audio given magnitude and phase. Of shape (num_batch, num_samples)
        """
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase     = torch.atan2(imag_part.data, real_part.data)

        recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)
        inverse_transform         = F.conv_transpose1d(recombine_magnitude_phase, self.inverse_basis, stride=self.hop_length, padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount:]
        inverse_transform = inverse_transform[..., :len(inverse_transform)-self.pad_amount]
        inverse_transform = inverse_transform.squeeze(1)
        return inverse_transform


    def forward(self, real_part, imag_part):
        inverse_transform = self.inverse(real_part, imag_part)
        return inverse_transform