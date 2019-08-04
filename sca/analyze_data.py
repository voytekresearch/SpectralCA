import sca

# imports
import os
import numpy as np
import scipy as sp
import scipy.io as io

if __name__ == __main__:
    main()

def main():
    run_kjm_analysis()
    

def run_kjm_analysis():

    def get_trial_info(cue_sig, move_sig, L_win=1000, L_lag=2000, alpha=0.95):
        # stim_info vector: [trial onset, movement onset, trial type, RT, premove duration, move duration]
        trial_init = np.where(np.diff((cue_sig>0).astype(int))==1)[0]+1
        move_init = np.where(np.diff((move_sig>0).astype(int))==1)[0]+1
        trial_info = []
        for i, tr_onset in enumerate(trial_init):
            # pre-movement length
            L_premove = np.sum(move_sig[tr_onset-L_win:tr_onset]<=0)
            # find corresponding movement onset time
            try:
                # if this fails, it means there was no response on the last trial
                mv_onset = move_init[np.where(move_init>tr_onset)[0][0]]
                L_move = np.sum(move_sig[mv_onset:mv_onset+L_win]>0)
                # if 3 conditions are satisfied, add trial
                if L_premove/L_win>alpha and L_move/L_win>alpha and (mv_onset-tr_onset)<L_lag:
                    RT = mv_onset-tr_onset
                    trial_info.append([tr_onset, mv_onset, move_sig[mv_onset], RT, L_premove, L_move])
            except:
                print('Skipped last trial.')
        return np.array(trial_info)

    def sca_kjm(data_path, subj, fs, analysis_param):
        # load data & trial info
        stim_info = io.loadmat(data_path+'data/'+subj+'/'+subj+'_stim.mat', squeeze_me=True)
        data = io.loadmat(data_path+'data/'+subj+'/'+subj+'_fingerflex.mat', squeeze_me=True)
        ecog = data['data'].T

        # set parameters
        win_len = int(fs)
        num_chan = ecog.shape[0]
        max_freq = analysis_param['max_freq']

        # sort out trial info
        trial_info = get_trial_info(data['cue'], stim_info['stim'], L_win=win_len, L_lag=1500, alpha=0.9)
        num_trials = trial_info.shape[0]
        print('%i included trials.'%num_trials)

        if num_trials>20:
            # pre-initialize arrays
            psd_precomp = np.zeros((num_chan, max_freq, num_trials, 2))

            # precompute trial PSDs
            for tr in range(num_trials):
                tr_onset = trial_info[tr,0]
                mv_onset = trial_info[tr,1]
                # pre-movement
                f_axis, t_axis, spg = sp.signal.spectrogram(ecog[:,tr_onset-win_len:tr_onset],fs=fs,nperseg=win_len)
                p_ = np.mean(spg,axis=-1)
                psd_precomp[:,:,tr,0] = p_[:,:max_freq]
                # movement
                f_axis, t_axis, spg = sp.signal.spectrogram(ecog[:,mv_onset:mv_onset+win_len],fs=fs,nperseg=win_len)
                p_ = np.mean(spg,axis=-1)
                psd_precomp[:,:,tr,1] = p_[:,:max_freq]

            sca_all = []
            for i in range(2):
                mov_sca = sca.SCA(analysis_param)
                mov_sca.populate_fourier_data(psd_precomp[:,:,:,i], fs, f_axis)
                mov_sca.compute_psd()
                mov_sca.compute_scv()
                sca_all.append(mov_sca)

            # return the pre- and during-movement sca data, and trial info
            return sca_all, trial_info
        else:
            return None, None


    # define data folder
    data_path = '/Users/ldliao/Research/Data/ECoG_KJM/digit/'
    saveout_path = '/Users/ldliao/Research/SpectralCA/results/kjm_digits/'
    save_files = ['pre','move','whole']
    subjs = ['bp', 'cc', 'ht', 'jc', 'jp', 'mv', 'wc', 'wm', 'zt']

    # electrode location
    elec_def = [('0', 'dorsal_M1'),
                ('1', 'dorsal_S1'),
                ('2', 'ventral_S1+M1'),
                ('3', 'frontal_(non-R)'),
                ('4', 'parietal_(non-R)'),
                ('5', 'temporal'),
                ('6', 'occipital')]

    fs = 1000.
    # sca params
    analysis_param = {'nperseg': 1000,
                     'noverlap': 0,
                     'spg_outlierpct': 2.,
                     'max_freq':200}

    for subj in subjs:
        print(subj)
        # get the trial-separated sca
        sca_all, trial_info = sca_kjm(data_path,subj,fs,analysis_param)
        if sca_all is not None:
            # get sca for whole recording
            data = io.loadmat(data_path+'data/'+subj+'/'+subj+'_fingerflex.mat', squeeze_me=True)
            sca_all.append(sca.SCA(analysis_param))
            sca_all[-1].populate_ts_data(data['data'].T, fs)
            sca_all[-1].compute_all_spectral()

            # make dir
            subj_path = saveout_path+subj+'/'
            if not os.path.isdir(subj_path):
                os.mkdir(subj_path)

            print('Computing exponential KS test...')
            for ind, sc in enumerate(sca_all):
                # compute fit
                sc.compute_KS_expfit()
                # save channel labels
                sc.chan_labels = ['chan_' + key + '_' + val for (key, val) in elec_def]
                print(ind)
                sc.cross_freq_corr()
                # save out sca
                sc.save_spec_vars(subj_path+save_files[ind]+'.npz', save_spg=True)

            # save trial info and plot contrast
            np.savez(subj_path+'trial_info.npz', trial_info=trial_info, elec_regions=data['elec_regions'])
        else:
            print('Skipped subject.')
