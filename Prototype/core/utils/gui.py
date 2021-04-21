from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
import pandas as pd
import seaborn as sn
from scipy.interpolate import make_interp_spline, BSpline

"""
Comparative plot of original signal and processed signal
"""
def plot_audio(wave_file,processed_file,sr):
    t = np.arange(wave_file.size)/ sr
    t2 = np.arange(processed_file.size)/ sr

    _,axs = plt.subplots(4,1)
    plt.sca(axs[0])
    plt.title("Signal and processed signal")
    plt.plot(t,wave_file,color='c',LineWidth=1.5,label='Noisy')
    plt.xlim(t[0],t[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.specgram(wave_file, NFFT=1024, Fs=sr, noverlap=900)
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.sca(axs[2])
    plt.plot(t2,processed_file,color='k',LineWidth=1.5,label='Clean')
    plt.xlim(t2[0],t2[-1])
    plt.legend()
    
    plt.sca(axs[3])
    plt.specgram(processed_file, NFFT=1024, Fs=sr, noverlap=900)
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.show()

"""
This function plot an Audio object.
"""
def plot_spectrum(spectrum,sample_rate):
    plt.figure(1)
    plt.specgram(spectrum, NFFT=1024, Fs=sample_rate, noverlap=900)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

"""
Plot spectro-data(librosa)
"""
def plot_librosa_spectrum(signal_stft):
    signal_stft_db = librosa.amplitude_to_db(np.abs(signal_stft), ref=np.max)
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig, ax = plt.subplots()
    img = librosa.display.specshow(signal_stft_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Now with labeled axes!')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

"""
Plot ODF function
"""
def plot_odf(name,signal,signal_proc,sr,onsets,hfc):
    so_time = np.arange(signal.size)/ sr
    sp_time = np.arange(signal_proc.size)/ sr
    onsets_time = np.arange(len(onsets))/ sr
    hfc_time = np.arange(len(hfc))/ sr
    
    plt.figure(1)
    plot_a = plt.subplot(411)
    plot_a.set_title(name)
    plot_a.plot(so_time,signal,color="k")
    plot_a.set_xlabel('Time')
    plot_a.set_ylabel('Energy')

    plot_a = plt.subplot(412)
    plot_a.plot(sp_time,signal_proc,color="y")
    plot_a.set_xlabel('Time')
    plot_a.set_ylabel('Energy')
    
    plot_c= plt.subplot(413)
    x_new = np.linspace(hfc_time.min(),hfc_time.max(),300)
    spl = make_interp_spline(hfc_time, hfc, k=3)  # type: BSpline
    hfc_smooth = spl(x_new)
    # plot_c.plot(x_new,hfc_smooth,color="b")
    plot_c.plot(hfc_time,hfc,color="b")
    plot_c.set_xlabel('Time')
    plot_c.set_ylabel('HFC')

    plot_c= plt.subplot(414)
    plot_c.plot(onsets_time,onsets,color="r")
    plot_c.set_xlabel('Time')
    plot_c.set_ylabel('Onset')
    
    plt.show()

"""
Plot De-noising fft comparative
"""
def plot_fft(signal,denoised_signal,sr,PSD,PSDClean,n):
    dt  = 0.001
    t  = np.arange(signal.size)/ sr
    fftfreq = (1/(dt*n))*np.arange(n)
    L = np.arange(1,np.floor(n/2),dtype="int")

    fig,axs = plt.subplots(2,1)
    
    plt.sca(axs[0])
    plt.plot(t,signal,color='c',LineWidth=1.5,label='Noisy')
    plt.plot(t,denoised_signal,color='k',LineWidth=1.5,label='Clean')
    plt.xlim(t[0],t[-1])
    plt.legend()

    plt.sca(axs[1])
    plt.plot(fftfreq[L],PSD[L],color='c',LineWidth=2,label='Noisy')
    plt.plot(fftfreq[L],PSDClean[L],color='k',LineWidth=2,label='Clean')
    plt.xlim(fftfreq[L[0]],fftfreq[L[-1]])
    plt.legend()
    plt.show()

"""
Plot evaluation report (f-score, recall, precision)
"""
def plot_evaluation_report(report):
        df = pd.DataFrame(report).T
        df['support'] = df.support.apply(int)
        sn.heatmap(df, annot=True, annot_kws={"size": 16}) # font size
        plt.show()

"""
Plot confusion matrix
"""
def plot_confusion_matrix(cm):
        df_cm = pd.DataFrame(cm, ["Kick","HH","Snare"], ["Kick","HH","Snare"])
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
        plt.show()

"""
Plot dataset statistics
"""
def plot_dataset_statistics(class_type,data):
    data = np.transpose(np.array(data))
    class_type.append("Total")
    collabel=("Nº Sounds", "Total duration (s)", "Average duration (s)")
    rowlabel= class_type
    plt.axis('off')
    the_table = plt.table(cellText=data,colLabels=collabel,rowLabels=rowlabel,loc='center')

    plt.show()


"""
Plot activity detection groundtruth vs predicted
"""
def plot_ad_evaluation(signal,groundtruth,predicted,sr):

    #If signal size is higher than 3s plot only first 3s for better analisys
    if signal.size > 130000:
        custom_time = np.arange(130000)/ sr
        signal_time = custom_time
        groundtruth_time = custom_time
        predicted_time = custom_time
        signal_size = 130000
        groundtruth_size = 130000
        predicted_size = 130000
    else:
        signal_size = signal.size
        groundtruth_size = len(groundtruth)
        predicted_size = len(predicted)
        signal_time = np.arange(signal_size)/ sr
        groundtruth_time = np.arange(groundtruth_size)/ sr
        predicted_time = np.arange(predicted_size)/ sr


    _, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Activity detection (groundtruth vs predicted)")
    ax.plot(signal_time,abs(signal[0:signal_size]),color="b",label="signal",alpha=0.2)
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    ax.plot(groundtruth_time,groundtruth[0:groundtruth_size],color="k",label="groundtruth")
    ax.legend()


    ax.plot(predicted_time,predicted[0:predicted_size],color="r",label="predicted")
    ax.legend()

    plt.show()
