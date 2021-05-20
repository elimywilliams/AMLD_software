#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday July 28
@author: emilywilliams
"""

import yaml
import json
import os

with open('/Users/emilywilliams/OneDrive - Southern Cross Corp/amld_script/running_scripts/amld_config.yaml', 'r') as file:
    configure_settings = yaml.load(file, Loader=yaml.FullLoader)

raw_data_loc = configure_settings['raw_data_loc']
function_file_Loc = configure_settings['function_file_Loc']
results_folder_loc = configure_settings['results_folder_loc']
car_id = configure_settings['car_id']
threshold = str(configure_settings['threshold'])
time_thresh = str(configure_settings['time_thresh'])
initial_time_ignore = str(configure_settings['initial_time_ignore'])
min_elevated = str(configure_settings['min_elevated'])
shift = configure_settings['shift']
engineering = configure_settings['engineering']
aeris = configure_settings['aeris']
CSU = configure_settings['CSU']
agg = configure_settings['agg']
amld = configure_settings['AMLD_vehicle']

time_push = configure_settings['time_push']
back_obs_num = str(configure_settings['back_obs_num'])
max_car_speed = str(configure_settings['max_car_speed'])
min_car_speed = str(configure_settings['min_car_speed'])
baseline_percentile = str(configure_settings['baseline_percentile'])
buffer_distance = str(configure_settings['buffer_distance'])
r_min = str(configure_settings['r_min'])
whichPolygon = str(configure_settings['polygon'])

raw_data_dir = results_folder_loc + 'RawData/'
observed_peaks_dir = results_folder_loc + 'ObservedPeaks/'
filtered_observed_peaks_dir = results_folder_loc + 'FilteredObservedPeaks/'
final_results_dir = results_folder_loc + 'FinalShpFiles/'
shp_file_loc = final_results_dir + 'verifiedPKs.json'
processedFileLoc = results_folder_loc + 'ProcessedData/'
op_shp_file_loc = final_results_dir + 'OP_Final.json'
all_op_csv_loc = final_results_dir + 'overallPeaks.csv'
final_info_loc = final_results_dir + 'summaryInfo.csv'
final_main_csv_loc = final_results_dir + 'mainThing.csv'

config_folder = results_folder_loc + 'Configuration/'
config_info = config_folder + 'algorithm_settings.csv'
configExists = os.path.exists(config_info)



s1 = car_id
s2 = "Peaks_" + str(s1)
s3 = "Filtered" + str()

##################################
### IMPORTING NECESSARY MODULES
##################################

import sys

sys.path.insert(0, function_file_Loc)  # FINDING FUNCTIONS FOLDER TO IMPORT FROM
from amld_functions import unique, unIfInt, \
    intersect, verPk, estimate_emissions, \
    haversine, wt_time_Locs, sum_values, make_GEO, \
    make_GPD, summarize_dat, get_quadrant, calc_theta, \
    calc_bearing, process_raw_data_eng, str_list, \
    count_times, identify_peaks, filter_peaks, \
    pass_combine, summarize_data_2, add_odometer, \
    process_raw_data, process_raw_data_aeris, \
    identify_peaks_CSU, weighted_loc, nameFiles, calc_velocity, \
    check_lst, nanthing, print_results, save_results, dt_to_epoch, minread_to_date, \
    identify_peaks_amld,process_raw_data_amld,identify_peaks_amld_aeris,process_raw_data_aeris_maybe

import rtree, os, sys, datetime, time, math, numpy, csv, gzip, shutil, ast, swifter
from math import radians, sin, cos, sqrt, asin
import numpy as np
from numpy import log
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from datetime import datetime
# import pygeos

#### CREATING NECESSARY FOLDERS
addingFiles = False
foldList = [results_folder_loc,raw_data_dir, observed_peaks_dir, filtered_observed_peaks_dir, final_results_dir,
            processedFileLoc,config_folder]
for x in foldList:
    if os.path.isdir(x) == False:
        try:
            os.mkdir(x)
        except OSError:
            print("Creation of the directory %s failed" % x)
        else:
            print("Successfully created the directory %s " % x)

### MOVING RAW FILES TO THE RAW DATA FILE FOLDER
for file in os.listdir(raw_data_loc):
    if file.endswith(".txt") and not agg:
        shutil.move(raw_data_loc + '/' + file, raw_data_dir)
    elif file.endswith('.csv') and amld:
        shutil.move(raw_data_loc + '/' + file, raw_data_dir)
    elif file.endswith('.csv') and agg:
        shutil.move(raw_data_loc + '/' + file, raw_data_dir)

########################################################################################

##### THIS PORTION OF THE CODE ALLOWS US TO ITERATIVELY ADD IN MORE DATA
#        PUT THE NEW TEXT FILES INTO THE OVERALL FOLDER AND IT WILL DO THE REST
if not agg and not amld:
    raw_txts = pd.DataFrame(os.listdir(raw_data_dir)).loc[
        pd.DataFrame(os.listdir(raw_data_dir))[0].str.endswith('.txt')]
if not agg and amld:
    raw_txts = pd.DataFrame(os.listdir(raw_data_dir)).loc[
        pd.DataFrame(os.listdir(raw_data_dir))[0].str.endswith('.csv')]
elif agg:
    raw_txts = pd.DataFrame(os.listdir(raw_data_dir)).loc[
        pd.DataFrame(os.listdir(raw_data_dir))[0].str.endswith('.csv')]

### DON'T ADD NEW FILE WITH PRE-EXISTING DATE [NEED TO WRITE CODE TO DEAL WITH THAT]
to_analyse, to_identify, to_filter = [[] for _ in range(3)]
fileDates = []

if os.path.exists(final_info_loc):
    analysed = pd.read_csv(final_info_loc)
    for index, row in raw_txts.reset_index().iterrows():
        text = raw_txts[0].iloc[index]
        if analysed[analysed['FILENAME'].astype(str).str.contains(text)].shape[0] < 1:
            to_analyse.append(text)
            to_identify.append(s1 + '_20' + text[11:17] + '_dat.csv')
            to_filter.append(s2 + '_20' + text[11:17] + '.csv')
elif not os.path.exists(final_info_loc):
    for index, row in raw_txts.reset_index().iterrows():
        text = raw_txts[0].iloc[index]
        to_analyse.append(text)
        if not agg and not amld:
            to_identify.append(s1 + '_20' + text[11:17] + '_dat.csv')
            to_filter.append(s2 + '_20' + text[11:17] + '.csv')
        elif agg and not amld:
            firstdate = datetime.fromtimestamp(pd.read_csv(raw_data_loc + '/RawData/' + text).timestamp[0]).strftime(
                '%Y%m%d')
            fileDates.append(firstdate)
            to_identify.append(s1 + '_' + firstdate + '_dat.csv')
            to_filter.append(s2 + '_' + firstdate + '.csv')
        elif amld:
            firstdate = str('20') + text[-17:-4]
            fileDates.append(firstdate)
            to_identify.append(s1 + '_' + firstdate + '_dat.csv')
            to_filter.append(s2 + '_' + firstdate + '.csv')

to_identify_unique = list(np.unique(np.array(to_identify)))


if not configExists:
    cLog = open(config_info, 'w')
    cLog.write('Algorithm Settings:' + '\n' )
    final = ''
    for key, val in configure_settings.items():
        woo = str(key) + ':' + str(val) + ' \n '
        final = final + str(woo)
    cLog.write(final)
    cLog.close()


##### START OF THE ALGORITHM
start = time.time()

if __name__ == '__main__':
    date_list = []
    x1 = ""
    count = 0
    for index, file in enumerate(to_analyse):
        # print (file)
        continue_analysis =  ((amld and file.endswith('.csv')) or (file.endswith(".txt")) )
        if continue_analysis:
            if (amld and file.endswith('.csv')):
                dateF = str('20') + file[-17:-4]
            elif file.endswith(".txt"):
                if not agg:
                    dateF = file[11:17]
                elif agg:
                    dateF = fileDates[index]
            if dateF in set(date_list):
                bFirst = False
            if dateF not in set(date_list):
                bFirst = True
                date_list.append(dateF)

            x1 = file[:10]
            x_date = dateF

            if engineering:
                the_result = process_raw_data_eng(car_id, x_date, raw_data_dir, file, bFirst, 1, processedFileLoc,
                                                  initial_time_ignore, shift, max_car_speed, min_car_speed)
            elif not engineering:
                if aeris:
                    the_result = process_raw_data_aeris(car_id, x_date, raw_data_dir, file, bFirst, 1, processedFileLoc,
                                                        initial_time_ignore, shift, max_car_speed, min_car_speed)
                elif not aeris and not amld:
                    the_result = process_raw_data(car_id, x_date, raw_data_dir, file, bFirst, 1, processedFileLoc,
                                                  initial_time_ignore, shift, max_car_speed, min_car_speed)
                elif not aeris and amld:
                    process_raw_data_amld(car_id, x_date, raw_data_dir, file, bFirst, 1, processedFileLoc,
                                          initial_time_ignore, shift, max_car_speed, min_car_speed)
            print(file)
            count = count + 1

# identify the peaks
if __name__ == '__main__':
    for index, file in enumerate(to_identify_unique):
        if file.startswith(s1) and file.endswith("dat.csv"):
            x_date = file[len(car_id) + 1:len(car_id) + 9]
            if amld:
                x_date = file[len(car_id) + 1:len(car_id) + 16]
            if CSU:
                the_result = identify_peaks_CSU(car_id, x_date, processedFileLoc, file, observed_peaks_dir,
                                                processedFileLoc,
                                                threshold, r_min, time_thresh, min_elevated,
                                                back_obs_num, baseline_percentile)
            elif not CSU and not amld:
                the_result = identify_peaks(car_id, x_date, processedFileLoc, file, observed_peaks_dir,
                                            processedFileLoc, engineering, threshold, r_min, time_thresh, min_elevated,
                                            back_obs_num, baseline_percentile)
            elif amld and not CSU:
                the_result = identify_peaks_amld(car_id, x_date, processedFileLoc, file, observed_peaks_dir,
                                            processedFileLoc, engineering, threshold, r_min, time_thresh, min_elevated,
                                            back_obs_num, baseline_percentile)

# filter the peaks
if __name__ == '__main__':
    index = 0
    numproc = 0
    for file in os.listdir(observed_peaks_dir):
        if file.startswith(s2) and (file.endswith('.csv') and not file.endswith('info.csv')):
            file_loc = observed_peaks_dir + file
            csv_loc = observed_peaks_dir + file[:-4] + '_info.csv'
            non_empty = False
            if pd.read_csv(file_loc).size != 0:
                print(file)
                index += 1
                non_empty = True
                x_date = file[len(car_id) + 7: len(car_id) + 15]
                if amld:
                    x_date = file[len(car_id) + 7: len(car_id) + 22]
                filter_peaks(car_id, x_date, observed_peaks_dir, file, filtered_observed_peaks_dir,
                             buffer=buffer_distance, whichpass=index)

if not os.path.exists(final_main_csv_loc):
    toCombine = os.listdir(filtered_observed_peaks_dir)
    toCombineList = []
    index = 0
    for file in toCombine:
        if file.startswith('Filtered') and file.endswith('.csv') and not file.endswith('info.csv'):
            toCombineList.append(file)
    index = 1
    for file in toCombineList:
        if index == 1:
            loc = filtered_observed_peaks_dir + file
            mainInfo = pd.read_csv(filtered_observed_peaks_dir + file[:-4] + '_info.csv')
            mainThing = (pd.read_csv(loc))
        elif index != 1:
            loc2 = filtered_observed_peaks_dir + file
            secThing = pd.read_csv(loc2)
            secInfo = pd.read_csv(filtered_observed_peaks_dir + file[:-4] + '_info.csv')
            woo = pass_combine(mainThing, secThing, car_id, buffer_distance)
            mainInfo = pd.concat([secInfo, mainInfo])
            mainThing = woo.copy()
        index = index + 1
        print(file)
    savemain = mainThing.copy()
    mainThing = mainThing.copy().reset_index(drop=True)
    mainThing['issue'] = mainThing.apply(lambda x: 'Issue' if x.recombine[1] == '.' else 'No Issue', axis=1)
    mainThing1 = mainThing.loc[mainThing.issue == 'No Issue', :].reset_index(drop=True)

    mainThing1['numtimes'] = mainThing1.apply(lambda x: count_times(x.recombine, car_id), axis=1)
    mainThing1['verified'] = mainThing1.apply(lambda x: True if x.numtimes > 1 else False, axis=1)

elif os.path.exists(final_main_csv_loc):
    addingFiles = True
    toCombine = list(map(lambda x: 'FilteredPeaks_' + x[6:], to_filter))
    mainThing = pd.read_csv(final_main_csv_loc)
    mainInfo = pd.read_csv(final_info_loc)
    curVP = mainThing.loc[mainThing.numtimes > 1, :].min_read.drop_duplicates().shape[0]
    curOP = mainThing.min_read.drop_duplicates().shape[0]
    for file in toCombine:
        loc = filtered_observed_peaks_dir + file
        if os.path.exists(loc):
            secThing = pd.read_csv(loc)
            mainThing = pass_combine(mainThing, secThing, car_id, buffer_distance)
            mainInfo = pd.concat([mainInfo, pd.read_csv(filtered_observed_peaks_dir + file[:-4] + '_info.csv')])
            print(file)
    mainThing = mainThing.copy().reset_index(drop=True)
    mainThing['issue'] = mainThing.apply(lambda x: 'Issue' if x.recombine[1] == '.' else 'No Issue', axis=1)
    mainThing1 = mainThing.loc[mainThing.issue == 'No Issue',:].reset_index(drop=True)

    mainThing1 = mainThing1.loc[mainThing1.OP_NUM!='foco_analysis_1614733528.0',:]
    mainThing1['numtimes'] = mainThing1.apply(lambda x: count_times(x.recombine, car_id), axis=1)
    mainThing1['verified'] = mainThing1.apply(lambda x: True if x.numtimes > 1 else False, axis=1)
    mainThing1['OB_CH4_ENH'] = mainThing1.apply(lambda x: x.OB_CH4 - x.OB_CH4_BASELINE, axis=1)
    mainThing1['OB_C2H6_ENH'] = mainThing1.apply(lambda x: x.OB_C2H6 - x.OB_C2H6_BASELINE, axis=1)

save_results(mainInfo, mainThing1, final_info_loc, final_main_csv_loc, shp_file_loc, op_shp_file_loc, all_op_csv_loc,
             threshold, car_id)
print_results(addingFiles, to_filter, threshold, baseline_percentile, back_obs_num, min_car_speed, max_car_speed,
              final_results_dir,
              start, mainThing1)
