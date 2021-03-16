from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
import pandas as pd
import seaborn as sn

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
def plot_odf(signal,sr,onsets):
    e_time = np.arange(signal.size)/ sr
    o_time = np.arange(len(onsets))/ sr
    
    plt.figure(1)
    plot_a = plt.subplot(211)
    plot_a.set_title("Onsets")
    plot_a.plot(e_time,signal,color="k")
    plot_a.set_xlabel('Time')
    plot_a.set_ylabel('Energy')
    
    plot_c= plt.subplot(212)
    plot_c.plot(o_time,onsets,color="r")
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