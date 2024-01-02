import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import resample, correlate, find_peaks
from scipy.interpolate import interp1d


ticks_per_pixel = 0.3 # Should recalibrate this!
microns_per_pixel = 1/18.28
ticks_per_micron = ticks_per_pixel/microns_per_pixel

kB = 1.38046e-23
# L0 = 8.3e-6 # 8.3 microns long
L0 = 5.4e-6#8.3e-6/0.5*1/3 # 8.3 microns long
T = 273.15 + 21 

"""
This script contains useful functions for analyzing the data from the optical tweezers setup.
Can help with correlating the data from the PSDs and the camera.

"""

def fixed_window_avg(data, window_size):
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    
    averages = []
    for i in range(0, len(data), window_size):
        window_data = data[i:i + window_size]
        avg = sum(window_data) / len(window_data)
        averages.append(avg)
        
    return np.array(averages)

def rolling_average(arr, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(arr, weights, mode='valid')

def get_speed(data, axis, offset=10,ticks_per_micron=5.484):
    """
    Extracts areas with movement speed greater than a certain threshold
    """
    key = "Motor_" + axis + "_pos"
    y = data[key]/ticks_per_micron
    motor_time = (data['Motor time']-data['Motor time'][0])/1e6
    speed = (y[offset:]-y[:-offset])/(motor_time[:-offset]-motor_time[offset:]) # Speed in microns/s
    return speed

def find_movement_areas(data, axis, threshold = 200, offset=10, positive=True,margin=100):
    """
    Extracts areas with movement speed greater than a certain threshold
    """
    key = "Motor_" + axis + "_pos"
    y = data[key]/ticks_per_micron
    motor_time = (data['Motor time']-data['Motor time'][0])/1e6
    speed = (y[offset:]-y[:-offset])/(motor_time[:-offset]-motor_time[offset:]) # Speed in microns/s
    
    #return speed
    _, move_indices = extract_high_rising_indices(speed, positive, threshold=threshold, margin=margin)
    
    # Extract the mean forces in positive and negative direction
    mean_force_A_x = []
    mean_force_B_x = []

    mean_force_A_y = []
    mean_force_B_y = []
    mean_speed = []
    
    for index_pair in move_indices:
        mean_speed.append(np.mean(speed[int(index_pair[0]):int(index_pair[1])]))

        start = int(index_pair[0]*14)
        stop = int(index_pair[1]*14)
        
        mean_force_A_x.append(np.mean(data['PSD_A_F_X'][start:stop]))
        mean_force_B_x.append(np.mean(data['PSD_B_F_X'][start:stop]))

        mean_force_A_y.append(np.mean(data['PSD_A_F_Y'][start:stop]))
        mean_force_B_y.append(np.mean(data['PSD_B_F_Y'][start:stop]))
    ret = np.mean(mean_force_A_x)
    return mean_speed, mean_force_A_x, mean_force_B_x, mean_force_A_y, mean_force_B_y

    
def extract_high_rising_indices(array, positive=True, threshold=200, margin=0):
    """
    
    """
    rising_indices = []
    falling_indices = []
    total_indices = []
    if not positive:
        array = -array
    high = False
    
    for idx, val in enumerate(array):
        if val>threshold and high == False:
            #rising_indices.append(idx)
            total_indices.append(idx+margin) 
            high = True
        if val<threshold and high == True:
            if idx - margin > total_indices[-1]:
                total_indices.append(idx-margin) 
            else:
                total_indices.pop()
            high = False
    total_indices = np.array(total_indices)
    indices_2d = np.zeros([int(len(total_indices)/2),2])
    indices_2d[:,0] = total_indices[0::2] # Rising
    indices_2d[:,1] = total_indices[1::2] # falling
    return total_indices, np.array(indices_2d)

def extract_avg_area(data, total_indices, key, avg_dist=500,margin=100,offset=70):
    """
    Extracts the average area around the given indices
    """
    avg_array_before = []
    for index_pair in total_indices:
        start = int((index_pair[0]-margin-avg_dist) *14)
        stop = int(index_pair[0]-margin)*14 #int(move_indices[0][1]*14)
        avg_array_before.append(np.mean(data[key][start:stop]))

    avg_array_after = []
    for index_pair in total_indices:
        start = int((index_pair[1]+margin+offset) *14)
        stop = int(index_pair[1]+margin+avg_dist)*14
        avg_array_after.append(np.mean(data[key][start:stop]))
    return avg_array_before, avg_array_after

def Calculate_PSD_Readings(data, axis, threshold = 200, offset=10, positive=True,margin=100,ticks_per_micron=5.484):
    """
    Extracts areas with movement speed greater than a certain threshold and uses this to calculate the PSD readings.
    """
    speed = get_speed(data, axis, offset=10,ticks_per_micron=ticks_per_micron)
    
    #return speed
    _, move_indices = extract_high_rising_indices(speed, positive, threshold=threshold, margin=margin)
    
    # Extract the mean forces in positive and negative direction
    mean_force_A_x = []
    mean_force_B_x = []

    mean_force_A_y = []
    mean_force_B_y = []
    mean_speed = []
    
    for index_pair in move_indices:
        mean_speed.append(np.mean(speed[int(index_pair[0]):int(index_pair[1])]))

        start = int(index_pair[0]*14)
        stop = int(index_pair[1]*14)
        
        mean_force_A_x.append(np.mean(data['PSD_A_F_X'][start:stop]))
        mean_force_B_x.append(np.mean(data['PSD_B_F_X'][start:stop]))

        mean_force_A_y.append(np.mean(data['PSD_A_F_Y'][start:stop]))
        mean_force_B_y.append(np.mean(data['PSD_B_F_Y'][start:stop]))
    #Extract averages around the movement areas
    AFX_before, AFX_after = extract_avg_area(data, move_indices, 'PSD_A_F_X', avg_dist=500,margin=margin,offset=60)
    AFY_before, AFY_after = extract_avg_area(data, move_indices, 'PSD_A_F_Y', avg_dist=500,margin=margin,offset=60)
    BFX_before, BFX_after = extract_avg_area(data, move_indices, 'PSD_B_F_X', avg_dist=500,margin=margin,offset=60)
    BFY_before, BFY_after = extract_avg_area(data, move_indices, 'PSD_B_F_Y', avg_dist=500,margin=margin,offset=60)
    # Put into arrays
    AFX = [mean_force_A_x, AFX_before, AFX_after]
    AFY = [mean_force_A_y, AFY_before, AFY_after]
    BFX = [mean_force_B_x, BFX_before, BFX_after]
    BFY = [mean_force_B_y, BFY_before, BFY_after]

    return mean_speed, AFX, BFX, AFY, BFY

def calc_force(speeds,eta=0.9795e-3,a=2.12e-6):
    """
    Calculates the force from the speed.
    eta = viscocity
    a = radius of the sphere
    """
    drag = 6 * np.pi*eta*a
    f = np.array(speeds)*1e-6*drag
    return f

def calc_sensitivity(data,axis, positive_dir,eta=0.9795e-3,a=2.12e-6,ticks_per_micron=5.484):
    analysis = Calculate_PSD_Readings(data,axis,offset=10, positive=positive_dir,ticks_per_micron=ticks_per_micron)
    force = calc_force(analysis[0], eta, a)
    if axis == 'x':
        i1 = 1
        i2 = 2
    else:
        i1 = 3
        i2 = 4
    A = np.array(analysis[i1][0])-(np.array(analysis[i1][1])+np.array(analysis[i1][2]))/2
    A_sense = force*1e12/A
    B = np.array(analysis[i2][0])-(np.array(analysis[i2][1])+np.array(analysis[i2][2]))/2
    B_sense = force*1e12/B
    return A_sense, B_sense

def calculate_sensitivities(data,eta=0.9795e-3,a=2.12e-6,ticks_per_micron=5.484):

    AX_pos, BX_pos = calc_sensitivity(data,'x',True,eta,a,ticks_per_micron)
    AX_neg, BX_neg = calc_sensitivity(data,'x',False,eta,a,ticks_per_micron)
    AY_pos, BY_pos = calc_sensitivity(data,'y',True,eta,a,ticks_per_micron)
    AY_neg, BY_neg = calc_sensitivity(data,'y',False,eta,a,ticks_per_micron)

    return AX_pos, BX_pos, AX_neg, BX_neg, AY_pos, BY_pos, AY_neg, BY_neg

def get_ticks_per_micron():
    ticks_per_pixel = 0.3 # Should recalibrate this!
    microns_per_pixel = 1/18.28
    ticks_per_micron = ticks_per_pixel/microns_per_pixel
    return 6.245

def calculate_position_scale_factor(position_microns,data,cam_shift_start=2, cam_shift_stop=2):
    cam_start, cam_stop = get_cam_move_lims(position_microns)
    cam_start -= cam_shift_start
    cam_stop += cam_shift_stop

    psd_start, psd_stop = get_limits_PSD(data)
    # TODO add the tilt to the analysis
    cam_data_pos = -resample_signal(position_microns[cam_start:cam_stop],psd_stop-psd_start)
    #cam_data_pos -= np.mean(cam_data_pos)
    psd_data_A = data['PSD_A_P_Y'][psd_start:psd_stop]/data['PSD_A_P_sum'][psd_start:psd_stop]

    psd_data_B = data['PSD_B_P_Y'][psd_start:psd_stop]/data['PSD_B_P_sum'][psd_start:psd_stop]


    psd_data_A -= np.mean(psd_data_A)
    psd_data_B -= np.mean(psd_data_B)

    scale_factor_A = np.dot(cam_data_pos, psd_data_A) / np.dot(psd_data_A, psd_data_A)
    scale_factor_B = np.dot(cam_data_pos, psd_data_B) / np.dot(psd_data_B, psd_data_B)
    return scale_factor_A, scale_factor_B


def resample_signal(y, target_length):
    """Resample y to match the length of target_length."""
    f = interp1d(np.linspace(0, 1, len(y)), y)
    return f(np.linspace(0, 1, target_length))

def best_shift(x1, x2):
    """Find the best shift to align x1 with x2."""
    # Compute cross-correlation
    cross_corr = correlate(x1, x2, mode='full')
    # The index of maximum correlation will give the best shift
    shift = cross_corr.argmax() - (len(x1) - 1)
    return shift

def get_limits_PSD(data):
    # Limits of the PSD signal
    start_dac = np.argmax(np.abs(data['dac_by']-data['dac_by'][0])>0)
    start_time = data['Motor time'][start_dac]
    start_psd_idx = np.argmin(np.abs(data['T_time'] - start_time))
    
    arr = np.abs(data['dac_by']-data['dac_by'][-2])>0
    last_idx = np.where(arr==1)[-1][-1]
    stop_time = data['Motor time'][last_idx]
    stop_psd_idx = np.argmin(np.abs(data['T_time'] - stop_time))
    return start_psd_idx, stop_psd_idx

def get_cam_move_lims(position_microns,first_lim=0.01, second_lim=0.1):
    start_cam = np.argmax(np.abs(position_microns-position_microns[0])>first_lim)

    arr = np.abs(position_microns-position_microns[-2])
    stop_cam = np.where(arr>second_lim)[0][-1]
    
    return start_cam, stop_cam
def find_fac(s1,s2):
    return np.sum(np.abs(s1-s2))

def find_unzipping_rezipping(Position, width=100):
    """
    Splits the force and position data into 
    Assumes that the P1 protocol was used for the zipping
    """
    # TODO cannot handle changing protocol during aquisition!
    # PRoblems if there is one peak without a matching "valley" somewhere in the middle.
    unzipping_force = []
    rezipping_force = []
    
    unzipping_distance = []
    rezipping_distance = []
    
    min_length = 100
    peaks, _ = find_peaks(Position, width=width)
    valleys, _ = find_peaks(-Position, width=width)
    
    idx = 0
    for peak in peaks:
        
        # Check that we have not gone trough the whole dataset
        if len(valleys) <= idx+1:
            return unzipping_force, rezipping_force

        # Look for peak
        if (valleys[idx] < peak) and (valleys[idx+1] > peak):
            # Save peak position
            unzipping_force.append([valleys[idx], peak])
            rezipping_force.append([peak, valleys[idx+1]])
            idx += 1
        elif valleys[idx] > peak:
            pass
        
    return unzipping_force, rezipping_force
"""
def find_increasing_decreasing_regions(signal):
    increasing_indices = []
    decreasing_indices = []
    
    for i in range(len(signal) - 1):
        if signal[i+1] > signal[i]:
            increasing_indices.append(i)
        elif signal[i+1] < signal[i]:
            decreasing_indices.append(i)
    
    return increasing_indices, decreasing_indices
"""
def find_increasing_decreasing_regions(signal):
    increasing_regions = []
    decreasing_regions = []
    
    current_increasing = []
    current_decreasing = []
    
    for i in range(len(signal) - 1):
        if signal[i+1] > signal[i]:
            current_increasing.append(i)
            if current_decreasing:
                decreasing_regions.append(current_decreasing)
                current_decreasing = []
        elif signal[i+1] < signal[i]:
            current_decreasing.append(i)
            if current_increasing:
                increasing_regions.append(current_increasing)
                current_increasing = []
                
    # Append remaining regions if they exist
    if current_increasing:
        increasing_regions.append(current_increasing)
    if current_decreasing:
        decreasing_regions.append(current_decreasing)
    
    return increasing_regions, decreasing_regions

def find_increase_decrease_of_data(data):
    tmp = resample_signal(data['dac_by'], len(data['PSD_B_P_sum']))
    return find_increasing_decreasing_regions(tmp)

def resample_signal(y, target_length):
    """Resample y to match the length of target_length."""
    f = interp1d(np.linspace(0, 1, len(y)), y)
    return f(np.linspace(0, 1, target_length))

def prepare_plot_data(data,Window_size=10,start=600_000,stop=700_000, shorten=False):
    psd_to_pos = [14.252,12.62]# [19,13.3]#[14.252,12.62]
    BFY_conversion = 0.0574201948018158 / 2
    AFY_conversion = 0.0513402835970939 / 2 
    analysis_func = fixed_window_avg if shorten else rolling_average

    X_data_A = analysis_func(data['PSD_A_P_Y'][start:stop]/data['PSD_A_P_sum'][start:stop], window_size=Window_size)
    Y_data_A = analysis_func(data['PSD_A_F_Y'][start:stop], window_size=Window_size)
    Y_data_A -= np.min(Y_data_A)
    Y_data_A *= AFY_conversion

    X_data_B = analysis_func(data['PSD_B_P_Y'][start:stop]/data['PSD_B_P_sum'][start:stop], window_size=Window_size)
    Y_data_B = analysis_func(data['PSD_B_F_Y'][start:stop], window_size=Window_size)
    Y_data_B -= np.min(Y_data_B)
    Y_data_B *= BFY_conversion

    X_data_A *= psd_to_pos[0]
    X_data_A -= np.min(X_data_A)
    X_data_B *= psd_to_pos[1]
    X_data_B -= np.min(X_data_B)
    X_data = np.array(X_data_A + X_data_B)/2
    Y_data = Y_data_A + Y_data_B
    return X_data, Y_data, X_data_A, Y_data_A, X_data_B, Y_data_B



def WLC_model(Lp,T,x,L0):
    # Most simple fitting model 
    # Parameters:
    # Lp - consitence length
    # x - extension (nm or m)
    # T - Temperature (K)
    # L0 - length of DNA
    f1 = 1.0 / (4 * ((1 - x / L0)**2))
    F = kB*T/Lp*(f1-1/4+x/L0)
    return F

def WLC_model_accurate(Lp,T,x,L0):
    # Parameters:
    # Lp - consitence length
    # x - extension (nm or m)
    # T - Temperature (K)
    # L0 - length of DNA
    # More accurate model on wikipedia, see https://en.wikipedia.org/wiki/Worm-like_chain
    f1 = 1.0 / (4 * ((1 - x / L0)**2))
    f1 += -1/4+x/L0
    
    alphas = [-0.5164228, -2.737418, 16.07497, -38.87607, 39.49944, -14.17718]
    for i, alpha in enumerate(alphas):
        f1 += alpha * ((x/L0)**(i+2))
    return kB*T/Lp*f1

def WLC_model_accurate_enthalpic(Lp,T,x,L0):
    L = (x/L0-F/K0)

    return

def WLC_model_p_fit(F,T,x,L0):
    # Parameters:
    # Lp - consitence length
    # x - extension (nm or m)
    # T - Temperature (K)
    # L0 - length of DNA
    f1 = 1.0 / (4 * ((1 - x / L0)**2))
    Lp = kB*T/F*(f1-1/4+x/L0)
    return Lp
"""
Here is an example script for analyzing the data from a measurement which includes both 

"""