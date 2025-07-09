import xarray as xr
import numpy as np
import os
import rasterio
from glob import glob
from tqdm import tqdm

def compute_topt_exceedance(
    crop,
    prefix,
    reference_tif_path,
    phen_start_path,
    phen_end_path,
    topt_path,
    temperature_folder_path,
    output_dir,
    scenario,
    start_year,
    end_year,
    window_length
):
    """
    Compute the first year when growing season mean temperature exceeds crop-specific Topt.
    
    Parameters
    ----------
    crop : str
        Crop name (e.g., "maize").
    prefix : str
        Prefix for output filenames (e.g., "SSP126").
    reference_tif_path : str
        Path to a reference GeoTIFF file for geospatial metadata.
    phen_start_path : str
        Path to crop phenology start date GeoTIFF (DOY).
    phen_end_path : str
        Path to crop phenology end date GeoTIFF (DOY).
    topt_path : str
        Path to crop Topt GeoTIFF.
    temperature_folder_path : str
        Directory containing NetCDF temperature files.
    output_dir : str
        Directory to save output GeoTIFFs.
    scenario : str
        Climate scenario identifier (e.g., "SSP126").
    start_year : int
        Start year for the analysis (e.g., 2020).
    end_year : int
        End year for the analysis (e.g., 2100).
    window_length : int
        Length of the moving time window (e.g., 20).
        
    Returns
    -------
    None
        Outputs GeoTIFF files recording first exceedance year for each grid cell.
    """
    print("Reading geospatial reference data...")
    with rasterio.open(reference_tif_path) as src:
        a = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile
        m, n = a.shape
    
    print("Reading phenology data...")
    phen_start = rasterio.open(phen_start_path).read(1)
    phen_end = rasterio.open(phen_end_path).read(1)
    
    print("Reading Topt data...")
    topt = rasterio.open(topt_path).read(1)
    
    print("Locating temperature files...")
    temperature_files = glob(os.path.join(temperature_folder_path, f'{scenario}*_8day_tmax_bc.nc'))
    
    # Read latitude/longitude from the first file
    first_nc_path = temperature_files[0]
    temperature_ds = xr.open_dataset(first_nc_path)
    temperature_data = temperature_ds['tmax'].isel(time=0)
    lats = temperature_data['lat'].values
    lons = temperature_data['lon'].values
    flipped_lats = lats[::-1]
    
    # Create DataArray for phenology
    phen_start_da = xr.DataArray(phen_start, coords={'lat': flipped_lats, 'lon': lons}, dims=('lat', 'lon'))
    phen_end_da = xr.DataArray(phen_end, coords={'lat': flipped_lats, 'lon': lons}, dims=('lat', 'lon'))
    
    for temperature_nc_path in temperature_files:
        print(f"Processing {temperature_nc_path}")
        modelname = temperature_nc_path.split(f'{scenario}_')[1].split('_8day')[0]
        temperature_ds = xr.open_dataset(temperature_nc_path)
        temperature_data = temperature_ds['tmax'].transpose('time', 'lat', 'lon')
        
        lats = temperature_data['lat'].values
        lons = temperature_data['lon'].values
        lat_order_nc = 1 if lats[1] > lats[0] else 0
        lon_order_nc = 1 if lons[1] > lons[0] else 0
        if not (lon_order_nc == 1 and lat_order_nc == 1):
            print("Warning: Latitude or longitude order is not ascending in NetCDF file.")
        
        result = np.full(phen_start.shape, np.nan)
        found_exceed_topt = np.zeros_like(result, dtype=bool)
        valid_mask = (phen_start > 0) & (phen_end > 0) & (topt > 0)
        
        for window_end_year in tqdm(range(start_year + window_length - 1, end_year + 1), desc="Processing Years"):
            window_start_year = window_end_year - window_length + 1
            print(f"Sliding window: {window_start_year}-{window_end_year}")
            
            window_temp_data = temperature_data.sel(
                time=slice(f'{window_start_year}-01-01', f'{window_end_year}-12-31'),
                lat=temperature_data['lat'][::-1]
            )
            
            doy = window_temp_data['time'].dt.dayofyear
            doy_expanded = doy.expand_dims({'lat': flipped_lats, 'lon': lons})
            phen_start_expanded = phen_start_da.expand_dims(dim={'time': window_temp_data['time']})
            phen_end_expanded = phen_end_da.expand_dims(dim={'time': window_temp_data['time']})
            
            mask = (
                ((phen_start_expanded < phen_end_expanded) &
                 ((doy_expanded >= phen_start_expanded) & (doy_expanded <= phen_end_expanded))) |
                ((phen_start_expanded >= phen_end_expanded) &
                 ((doy_expanded >= phen_start_expanded) | (doy_expanded <= phen_end_expanded)))
            )
            
            growing_season_temp = window_temp_data.where(mask)
            growing_season_mean_temp = growing_season_temp.mean(dim='time')
            
            exceed_topt = (growing_season_mean_temp > topt) & valid_mask & (~found_exceed_topt)
            result[exceed_topt] = window_end_year
            found_exceed_topt = found_exceed_topt | exceed_topt
            
            if found_exceed_topt.all():
                break
        
        result[valid_mask & ~found_exceed_topt] = end_year + 1
        
        output_filename = f'{prefix}_{modelname}_{crop}_0.tif'
        output_path = os.path.join(output_dir, output_filename)
        profile.update(dtype=rasterio.float32, count=1, transform=transform, crs=crs)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(result.astype(np.float32), 1)
        
        print(f"Result saved to {output_path}")
