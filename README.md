My implementation of EINV2 with modification to run on Kaggle Notebook, most of which is due to newer versions of libraries. Original Source code could be found at https://github.com/yinkalario/EIN-SELD

# Modification:
Adding in .sh scripts
```
   source activate ein
```
Change seld/learning/preprocess.py line 148 and seld/methods/ein_seld/training.py line 109 to (this problem could be found in Source code -> Issue but dont if you use environment.yml file to create environment)
```
   batch_x = batch_sample.batch_out_dict['waveform']
```
In seld/methods/ein_seld/utils, change SELD_evaluation_metrics_2019.py and SELD_evaluation_metrics_2020.py line 11 to 
```
   eps = np.finfo(float).eps
```  
this is because numpy ver 1.20 remove np.float

In seld/methods/ein_seld/utils, change stft.py line 164 to
```
   fft_window = librosa.util.pad_center(fft_window, size=n_fft)
```
line 86, 87, 134 change ```n``` to ```self.n```
