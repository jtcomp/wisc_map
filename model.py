import netCDF4
import numpy as np
import os
import datetime
import gc

import matplotlib.pyplot as plt
from netCDF4 import Dataset,num2date,date2num

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def debug_data(mat):
    plt.imshow(mat)
    plt.colorbar()
    plt.show()


def debug_snapshot(mat):
    plt.imshow(mat)
    plt.colorbar()
    plt.show()


def load_data(GCMs):
    # loop across years in 20-year increments
    for y in range(2030,2091,20):    # 2030,2091,20
        print("  Processing base year "+str(y))

        # create (large) arrays for all data within current set of years
        # loop through GCMS first and then years in current time window
        # (create large blank masked arrays more memory efficient?)
        num_models = len(GCMs)
        years = 20
        realizations = 3
        tmax_all_array = np.ma.empty([num_models*years*realizations, 365, 48, 62])
        tmax_means_array = np.ma.empty([num_models, 365, 48, 62])
        tmin_all_array = np.ma.empty([num_models*years*realizations, 365, 48, 62])
        tmin_means_array = np.ma.empty([num_models, 365, 48, 62])
        
        # indices for adding to blank arrays
        i = 0
        j = 0
        
        for gcm in GCMs:    
            print("    Processing GCM "+gcm)
            tmax_gcm_array = []
            tmin_gcm_array = []
            for yr in range(y-9,y+11):
                print("      Processing year "+str(yr))
                realizations = ["01","02","03"]
                # get data for all three "realizations" from each GCM
                for r in realizations:
                    nf = os.path.join(base_folder,m,gcm,"temp_"+r+"_"+str(yr)+".nc")
                    try:
                        n = netCDF4.Dataset(nf)
                    except:
                        # only happens if file is missing -- does happen for a couple of years in the 
                        # first 20-year window
                        q = 0    # does nothing
                    else:
                        # found file, now lets get tmin and tmax for this GCM realization
                        tmax = n.variables['tmax']
                        tmin = n.variables['tmin']
                        
                        # convert to F
                        tmax = (tmax[:,:,:] * 1.8) + 32
                        tmin = (tmin[:,:,:] * 1.8) + 32
                        
                        # get mask - to reapply below
                        mask = np.ma.getmask(tmax[0])
                        
                        # handle leap years -- average Feb 28 and 29
                        if tmax.shape[0] == 366:
                            # get and average Feb 28 and 29 for tmax
                            tmax_leap = tmax[58:60,:,:]
                            tmax_mean_leap = np.ma.mean(tmax_leap,axis=0)
                            tmax_part1 = np.ma.append(tmax[0:58,:,:],np.ma.expand_dims(tmax_mean_leap,axis=0),axis=0)
                            tmax = np.ma.append(tmax_part1,tmax[60:367,:,:],axis=0)
                            # ...and for tmin
                            tmin_leap = tmin[58:60,:,:]
                            tmin_mean_leap = np.ma.mean(tmin_leap,axis=0)
                            tmin_part1 = np.ma.append(tmin[0:58,:,:],np.ma.expand_dims(tmin_mean_leap,axis=0),axis=0)
                            tmin = np.ma.append(tmin_part1,tmin[60:367,:,:],axis=0)
                        
                        # add tmax array for gcm, year, realization to total array
                        tmax_all_array[i] = tmax[:,:,:]
                        tmin_all_array[i]= tmin[:,:,:]
                        
                        # add tmin and tmax array for gcm, year, realization to gcm-specific array
                        # (used below for inter-model standard deviation)
                        tmax_gcm_array.append(tmax[:,:,:])
                        tmin_gcm_array.append(tmin[:,:,:])
                        
                        i += 1
                        
                    # end realization loop
                # end year loop
            
            # calculate mean temperature for gcm over time window (for std calculation)
            tmax_model_mean = np.ma.mean(tmax_gcm_array,axis=0)
            tmin_model_mean = np.ma.mean(tmin_gcm_array,axis=0)

            # add this to an array of means (one for each gcm)
            tmax_means_array[j] = tmax_model_mean[:,:,:]
            tmin_means_array[j] = tmin_model_mean[:,:,:]
            
            j += 1

        yield tmax_all_array, tmin_all_array, tmax_means_array, tmin_means_array, mask, y, n
            

def process_write_data (tmax_all_array, tmin_all_array, tmax_means_array, tmin_means_array, mask, y, n):

    ##################################################
    ## calculate metrics for current 20-year window ##
    ##################################################


    # these are all some ungodly shape
    # (iter# x yr) (days) (y) (x)
    # mask is land
    [print(f"{mat[0]}:{mat[1].shape}") for mat in {
        "tmax_all_array":tmax_all_array,
        "tmin_all_array":tmin_all_array,
        "tmax_means_array":tmax_means_array,
        "tmin_means_array":tmin_means_array,
        "mask":mask
          }.items() ] 

    debug_data(np.flipud(tmax_all_array[0][0]))

    # calculate daily mean from total array    
    tmax_mean = np.ma.mean(tmax_all_array,axis=0)
    tmin_mean = np.ma.mean(tmin_all_array,axis=0)
    
    # convert/calculate monthly values from total array (in days)
    tmax_months_mean = np.ma.empty([12, 48, 62])
    tmin_months_mean = np.ma.empty([12, 48, 62])
    tmax_months_std = np.ma.empty([12, 48, 62])
    tmin_months_std = np.ma.empty([12, 48, 62])
    tmax_months_gt100 = np.ma.empty([12, 48, 62])
    tmax_months_gt95 = np.ma.empty([12, 48, 62])
    tmax_months_gt90 = np.ma.empty([12, 48, 62])
    tmax_months_comfort = np.ma.empty([12, 48, 62])
    tmin_months_gt70 = np.ma.empty([12, 48, 62])
    tmin_months_lt32 = np.ma.empty([12, 48, 62])
    tmin_months_lt0 = np.ma.empty([12, 48, 62])
    
    # index for julian/ordinal calendar (start/end of each month)
    julian = [0,31,59,90,120,151,181,212,243,273,304,334,366]
    
    # loop through months
    for month in range(1,13):
        # calculate mean values for each month, add to monthly value array
        print('Calc mean for index start '+str(julian[month-1])+' to '+str(julian[month]))
        tmax_month_mean = np.ma.mean(tmax_mean[(julian[month-1]):(julian[month])],axis=0)
        tmax_months_mean[month-1] = tmax_month_mean
        tmin_month_mean = np.ma.mean(tmin_mean[(julian[month-1]):(julian[month])],axis=0)
        tmin_months_mean[month-1] = tmin_month_mean
        
        # calculate inter-model standard deviations
        all_std_tmax_nparray = np.array(tmax_means_array)
        month_std_tmax_nparray = all_std_tmax_nparray[:,(julian[month-1]):(julian[month]),:,:]
        all_std_tmin_nparray = np.array(tmin_means_array)
        month_std_tmin_nparray = all_std_tmin_nparray[:,(julian[month-1]):(julian[month]),:,:]
        # loop over each model -- add to total monthly sample array
        for i in range(0, (np.shape(all_std_tmax_nparray)[0])):
            model_tmax_array = month_std_tmax_nparray[i]
            model_tmin_array = month_std_tmin_nparray[i]
            if i==0:
                concat_tmax_array = model_tmax_array
                concat_tmin_array = model_tmin_array
            else:
                concat_tmax_array = np.ma.concatenate((concat_tmax_array,model_tmax_array),axis=0)
                concat_tmin_array = np.ma.concatenate((concat_tmin_array,model_tmin_array),axis=0)
        tmax_std_month = np.ma.std(concat_tmax_array,axis=0)
        tmax_months_std[month-1] = np.ma.masked_array(tmax_std_month,mask)
        tmin_std_month = np.ma.std(concat_tmin_array,axis=0)
        tmin_months_std[month-1] = np.ma.masked_array(tmin_std_month,mask)
        
        # calc extreme estimates - tmax
        all_nparray = np.array(tmax_all_array)
        month_nparray = all_nparray[:,(julian[month-1]):(julian[month]),:,:]
        # get number of days in month and number of "samples"(needed for estimate formula)
        num_days = np.shape(month_nparray)[1]
        num_samples = np.shape(month_nparray)[0]*np.shape(month_nparray)[1]     
        
        # tmax extreme - tmax gt 100 F
        # get number of days meeting threshold (by iteration)
        extreme_sum_days = (month_nparray >= 100).sum(axis=0)
        # add number of days (by month)
        extreme_sum = extreme_sum_days.sum(axis=0)
        # estimate of number of monthly extreme days
        extreme_days = (extreme_sum / num_samples) * num_days
        # add to array
        tmax_months_gt100[month-1] = np.ma.masked_array(extreme_days,mask)
        
        # tmax extreme - tmax gt 95 F     
        # get number of days meeting threshold (by iteration)
        extreme_sum_days = (month_nparray >= 95).sum(axis=0)
        # add number of days (by month)
        extreme_sum = extreme_sum_days.sum(axis=0)
        # estimate of number of monthly extreme days
        extreme_days = (extreme_sum / num_samples) * num_days
        # add to array
        tmax_months_gt95[month-1] = np.ma.masked_array(extreme_days,mask)
        
        # tmax extreme - tmax gt 90 F     
        # get number of days meeting threshold (by iteration)
        extreme_sum_days = (month_nparray >= 90).sum(axis=0)
        # add number of days (by month)
        extreme_sum = extreme_sum_days.sum(axis=0)
        # estimate of number of monthly extreme days
        extreme_days = (extreme_sum / num_samples) * num_days
        # add to array
        tmax_months_gt90[month-1] = np.ma.masked_array(extreme_days,mask)
        
        # tmax extreme(?) - tmax gt 70 F and lt 85 F    
        # get number of days meeting threshold (by iteration)
        extreme_sum_days = ((month_nparray >= 70) & (month_nparray <= 85)).sum(axis=0)
        # add number of days (by month)
        extreme_sum = extreme_sum_days.sum(axis=0)
        # estimate of number of monthly extreme days
        extreme_days = (extreme_sum / num_samples) * num_days
        # add to array
        tmax_months_comfort[month-1] = np.ma.masked_array(extreme_days,mask)
        
        # calc extreme estimates - tmin
        all_nparray = np.array(tmin_all_array)
        month_nparray = all_nparray[:,(julian[month-1]):(julian[month]),:,:]
        # get number of days in month and number of "samples"(needed for estimate formula)
        num_days = np.shape(month_nparray)[1]
        num_samples = np.shape(month_nparray)[0]*np.shape(month_nparray)[1] 
        
        # tmin extreme - tmin gt 70 F     
        # get number of days meeting threshold (by iteration)
        extreme_sum_days = (month_nparray >= 70).sum(axis=0)
        # add number of days (by month)
        extreme_sum = extreme_sum_days.sum(axis=0)
        # estimate of number of monthly extreme days
        extreme_days = (extreme_sum / num_samples) * num_days
        # add to array
        tmin_months_gt70[month-1] = np.ma.masked_array(extreme_days,mask)
        
        # tmin extreme - tmin lt 32 F     
        # get number of days meeting threshold (by iteration)
        extreme_sum_days = (month_nparray <= 32).sum(axis=0)
        # add number of days (by month)
        extreme_sum = extreme_sum_days.sum(axis=0)
        # estimate of number of monthly extreme days
        extreme_days = (extreme_sum / num_samples) * num_days
        # add to array
        tmin_months_lt32[month-1] = np.ma.masked_array(extreme_days,mask)
        
        # tmin extreme - tmin lt 0 F     
        # get number of days meeting threshold (by iteration)
        extreme_sum_days = (month_nparray <= 0).sum(axis=0)
        # add number of days (by month)
        extreme_sum = extreme_sum_days.sum(axis=0)
        # estimate of number of monthly extreme days
        extreme_days = (extreme_sum / num_samples) * num_days
        # add to array
        tmin_months_lt0[month-1] = np.ma.masked_array(extreme_days,mask)
    
    # write netcdf files with results
    filename = "temp_"+m+"_"+str(y)+"_20yr_monthly.nc"
    newfile = os.path.join(out_folder,filename)
    ncfile = netCDF4.Dataset(newfile,mode='w',format='NETCDF4_CLASSIC')
    lat_dim = ncfile.createDimension('lat', 48)     # latitude axis
    lon_dim = ncfile.createDimension('lon', 62)    # longitude axis
    time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).

    ncfile.title='Aggregate monthly tmax and tmin values for WICCI downscaled climate data for all GCMs for '+m+' and 20-year window around year '+str(y)
    ncfile.subtitle="Data source: UW-Madison WICCI; Data aggregation: Eric Compas, compase@uww.edu"
    lat = ncfile.createVariable('lat', np.float64, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float64, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float64, ('time',))
    timeunits = 'days since '+str(y)+'-01-01'
    time.units = timeunits
    time.long_name = 'time'

    temp_tmax_mean = ncfile.createVariable('tmax_mean',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmax_mean.units = 'F' # Fahrenheit 
    temp_tmax_mean.standard_name = 'mean of maximum daily temperature (F) across 20-year window'
    temp_tmax_mean.missing_value = -32768
    
    temp_tmax_std = ncfile.createVariable('tmax_std',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmax_std.units = 'F' # Fahrenheit 
    temp_tmax_std.standard_name = 'standard deviation of maximum daily temperature (F) across 20-year window'
    temp_tmax_std.missing_value = -32768
    
    temp_tmin_mean = ncfile.createVariable('tmin_mean',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmin_mean.units = 'F' # Fahrenheit 
    temp_tmin_mean.standard_name = 'mean of minimum daily temperature (F) across 20-year window'
    temp_tmin_mean.missing_value = -32768

    temp_tmin_std = ncfile.createVariable('tmin_std',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmin_std.units = 'F' # Fahrenheit 
    temp_tmin_std.standard_name = 'standard deviation of minimum daily temperature (F) across 20-year window'
    temp_tmin_std.missing_value = -32768
    
    temp_tmax_gt100 = ncfile.createVariable('tmax_gt100',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmax_gt100.units = 'days' # number of days
    temp_tmax_gt100.standard_name = 'estimated number of days where maximum daily temperature equals or exceeds 100 F '+\
                                    'across 20-year window'
    temp_tmax_gt100.missing_value = -32768
    
    temp_tmax_gt95 = ncfile.createVariable('tmax_gt95',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmax_gt95.units = 'days' # number of days
    temp_tmax_gt95.standard_name = 'estimated number of days where maximum daily temperature equals or exceeds 95 F '+ \
                                    'across 20-year window'
    temp_tmax_gt95.missing_value = -32768
    
    temp_tmax_gt90 = ncfile.createVariable('tmax_gt90',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmax_gt90.units = 'days' # number of days
    temp_tmax_gt90.standard_name = 'estimated number of days where maximum daily temperature (C) equals or exceeds 90 F '+ \
                                    'across 20-year window'
    temp_tmax_gt90.missing_value = -32768
    
    temp_tmax_comfort = ncfile.createVariable('tmax_comfort',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmax_comfort.units = 'days' # number of days
    temp_tmax_comfort.standard_name = 'estimated number of "comfortable" days where maximum daily temperature is '+ \
                                        'greater than 70 F and less than 80 F across 20-year window'
    temp_tmax_comfort.missing_value = -32768
    
    temp_tmin_gt70 = ncfile.createVariable('tmin_gt70',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmin_gt70.units = 'days' # number of days
    temp_tmin_gt70.standard_name = 'estimated number of days where minimum daily temperature equals or exceeds ' + \
                                    '70 F across 20-year window'
    temp_tmin_gt70.missing_value = -32768
    
    temp_tmin_lt32 = ncfile.createVariable('tmin_lt32',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmin_lt32.units = 'days' # number of days
    temp_tmin_lt32.standard_name = 'estimated number of days where minimum daily temperature equals or is less than '+ \
                                    '32 F across 20-year window'
    temp_tmin_lt32.missing_value = -32768
    
    temp_tmin_lt0 = ncfile.createVariable('tmin_lt0',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    temp_tmin_lt0.units = 'days' # number of days
    temp_tmin_lt0.standard_name = 'estimated number of days where minimum daily temperature equals or is less than' + \
                                    '0 F across 20-year window'
    temp_tmin_lt0.missing_value = -32768
    

    # Write latitudes, longitudes
    # Note: the ":" is necessary in these "write" statements
    n_lat = n.variables['lat']
    n_lon = n.variables['lon']
    lat[:] = n_lat[:]
    lon[:] = n_lon[:]

    # write temp variables
    temp_tmax_mean[:,:,:] = tmax_months_mean
    temp_tmax_std[:,:,:] = tmax_months_std
    temp_tmin_mean[:,:,:] = tmin_months_mean
    temp_tmin_std[:,:,:] = tmin_months_std
    temp_tmax_gt100[:,:,:] = tmax_months_gt100
    temp_tmax_gt95[:,:,:] = tmax_months_gt95
    temp_tmax_gt90[:,:,:] = tmax_months_gt90
    temp_tmax_comfort[:,:,:] = tmax_months_comfort
    temp_tmin_gt70[:,:,:] = tmin_months_gt70
    temp_tmin_lt32[:,:,:] = tmin_months_lt32
    temp_tmin_lt0[:,:,:] = tmin_months_lt0

    # write time
    yystart = y
    nyears = 1
    ntime = 12
    print("Writing time. Year = "+str(y))
    datesout = [datetime.datetime(y,mm,15,0) for yy in range(yystart,yystart+nyears) for mm in range(1,13)]
    time[:] = date2num(datesout,timeunits)

    # close file
    ncfile.close()

    # report progress
    print("Wrote: "+filename)
    
    
    # clear some memory, invoke Python garbage collector
    del tmax_all_array
    del tmax_means_array
    del tmin_all_array
    del tmin_means_array
    del tmax_months_mean
    del tmin_months_mean
    del tmax_months_std
    del tmin_months_std
    del tmax_months_gt100
    del tmax_months_gt95
    del tmax_months_gt90
    del tmax_months_comfort
    del tmin_months_gt70
    del tmin_months_lt32
    del tmin_months_lt0
    del tmax_mean
    del tmin_mean
    gc.collect()
    print("Cleared some memory up...restarting loop")



 
print("running")

#base_folder = "Z:/Climate_Data"
base_folder = "./"
#base_folder = "/Users/ericcompas/Climate_Data"
if not os.path.isdir(base_folder):
    print("Base folder not valid")

#out_folder = "Z:/Climate_Data/aggregate_files"
out_folder = "./out"
#out_folder  = "/Users/ericcompas/Climate_Data/output"
if not os.path.isdir(out_folder):
    print("Out folder not valid")

# loop across models
#models = ["rcp45","rcp85"]
models = ["rcp85"]
for m in models:
    print("Processing climate scenario "+m)

    # get subfolders/GCMs for current model
    GCMs = os.listdir(os.path.join(base_folder,m))

    for data in load_data(GCMs):
        process_write_data(*data)
    #GCMs = GCMs[:3]  # for testing only