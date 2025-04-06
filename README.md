# insar-coreg-debug
Debugging InSAR.dev PyGMTSAR Coregistration Cell 2

# === 📦 Coregistration Setup (Final Revised Cell 2)
import os, shutil, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from datetime import datetime
from shapely.ops import unary_union
from types import MethodType
from pygmtsar import PRM
from pygmtsar.Stack_ps import Stack_ps
from pygmtsar.Stack_align import Stack_align
from pygmtsar.Stack_reframe_gmtsar import Stack_reframe_gmtsar

# === 📁 Paths
dem_file = "/mnt/e/InSARdev/Descending/DEM/srtm_jakarta.dem.wgs84"
prm_dir = "/mnt/e/InSARdev/Descending/PRM"
orbit_dir = "/mnt/e/InSARdev/Descending/Orbits"
workdir = "/mnt/e/InSARdev/Descending/align"
slc_root = "/mnt/e/InSARdev/Descending/SLC"

# === 🧠 Read PRM metadata
def get_prm_info(date_str):
    prm_path = os.path.join(prm_dir, f"{date_str}_F2.PRM")
    try:
        prm = PRM.from_file(prm_path)
        return (
            date_str,
            int(prm.sel("nl")),
            int(prm.sel("ns")),
            int(str(prm.get("swath") or "IW2")[-1]),
        )
    except:
        try:
            with open(prm_path) as f:
                lines = f.readlines()
            def extract(key):
                for line in lines:
                    if line.strip().startswith(key):
                        return line.split("=")[-1].strip()
                return None
            nl = int(extract("nl"))
            ns = int(extract("ns"))
            swath = extract("swath") or "IW2"
            subswath = int(str(swath)[-1])
            return (date_str, nl, ns, subswath)
        except:
            return None

# Build PRM DataFrame
prm_files = sorted([f for f in os.listdir(prm_dir) if f.endswith("_F2.PRM")])
raw_dates = [f.replace("_F2.PRM", "") for f in prm_files]
meta = [get_prm_info(d) for d in raw_dates]
meta_clean = [m for m in meta if m]
if not meta_clean:
    raise RuntimeError("❌ No valid PRMs found.")
df_prm = pd.DataFrame(meta_clean, columns=["date", "nl", "ns", "subswath"]).set_index("date")

# === 🔁 Build or update df with SLC path detection
try:
    df
except NameError:
    df = pd.DataFrame(index=df_prm.index)
    for col in ["datapath", "metapath", "noisepath", "calibpath"]:
        df[col] = ""

# === 🧠 Auto-detect SLC subfolders
def safe_autopath(date):
    ymd = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    folders = sorted(glob.glob(os.path.join(slc_root, f"*{ymd}*.SAFE")))
    if not folders:
        print(f"⚠️ No SAFE folder for {date}")
        return None, None, None, None
    base = folders[0]
    
    # Auto-discover files inside .SAFE structure
    annotation = glob.glob(os.path.join(base, "annotation", "*.xml"))
    measurement = glob.glob(os.path.join(base, "measurement", "*.tiff"))
    noise = glob.glob(os.path.join(base, "noise", "*.xml"))
    calibration = glob.glob(os.path.join(base, "calibration", "*.xml"))

    if not annotation or not measurement:
        print(f"⚠️ Missing annotation or measurement file in {base}")
        return None, None, None, None
    
    return measurement[0], annotation[0], noise[0] if noise else "", calibration[0] if calibration else ""

# === 📎 Merge all metadata
df_prm = df_prm[~df_prm.index.duplicated()]
df = df[~df.index.duplicated()]
valid_df = df_prm.join(df.drop(columns=["nl", "ns", "subswath"], errors="ignore"), how="left")

# === 🛰️ Setup Stack_ps
stack = Stack_ps()
stack.df = valid_df.copy()
stack.safes = valid_df.index.tolist()
stack.orbit = orbit_dir
stack.set_dem(dem_file)
stack.workdir = workdir
stack.reference = sorted(valid_df.index)[0]
os.makedirs(workdir, exist_ok=True)

# === 📂 Copy PRMs to align folder
for date in valid_df.index:
    src = os.path.join(prm_dir, f"{date}_F2.PRM")
    dst = os.path.join(workdir, f"{date}_F2.PRM")
    if os.path.exists(src):
        shutil.copy2(src, dst)

print("📤 All PRMs copied.")
print(f"✅ Using {len(valid_df)} PRM records.")

# === 🌍 Create GeoDataFrame
df = valid_df.copy()
df["geometry"] = None
df = gpd.GeoDataFrame(df, geometry="geometry")

# === 🔧 Patch DEM and alignment methods
def patched_get_dem(self):
    da = rioxarray.open_rasterio(self.dem_filename, masked=True)
    if "band" in da.dims: da = da.squeeze("band", drop=True)
    da = da.rename({"x": "lon", "y": "lat"})
    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da.rio.write_crs("EPSG:4326", inplace=True)
    if da.lat[0] > da.lat[-1]: da = da.sortby("lat")
    return da

def safe_get_topo_llt(self, subswath, degrees=0.01, debug=False):
    dem_area = self.get_dem()
    ny = int(np.round(degrees / dem_area.lat.diff("lat").values[0]))
    nx = int(np.round(degrees / dem_area.lon.diff("lon").values[0]))
    topo = dem_area.coarsen(lat=ny, lon=nx, boundary="pad").mean()
    lat_grid, lon_grid = np.meshgrid(topo.lat.values, topo.lon.values, indexing="ij")
    return np.stack((lat_grid, lon_grid, topo.values), axis=-1)

def patched_align_ref_subswath(self, subswath, date, debug=False):
    try:
        for k in ["metapath", "datapath"]:
            val = self.df.loc[date, k]
            if isinstance(val, list): self.df.loc[date, k] = val[0]
        if not self.df.loc[date, "metapath"] or not self.df.loc[date, "datapath"]:
            print(f"⚠️ Skipping {date}: missing metapath or datapath")
            return
        return Stack_align._align_ref_subswath(self, subswath, debug=debug)
    except Exception as e:
        print(f"⚠️ Path error for {date}: {e}")

def patched_align_rep_subswath(self, subswath, date, degrees=None, debug=False):
    try:
        if pd.isna(self.df.loc[date, "nl"]): return
        for k in ["metapath", "datapath"]:
            val = self.df.loc[date, k]
            if isinstance(val, list): self.df.loc[date, k] = val[0]
        if not self.df.loc[date, "metapath"] or not self.df.loc[date, "datapath"]:
            print(f"⚠️ Skipping {date}: missing metapath or datapath")
            return
        topo_llt = self._get_topo_llt(subswath, degrees=degrees)
        rep_prefix = self.get_subswath_prefix(subswath, date)
        slc_prefix = self.get_subswath_prefix(subswath, self.reference)
        self._make_s1a_tops(subswath, date=date)
        self.align_coarse(slc_prefix, rep_prefix, subswath, topo_llt)
        self.align_fine(slc_prefix, rep_prefix, subswath)
        self.corr_s1a(rep_prefix, subswath)
    except Exception as e:
        print(f"❌ Failed align_rep_subswath {date}: {e}")

# === ⛓️ Aligner with patched methods
class CustomStack(Stack_align, Stack_reframe_gmtsar): pass
aligner = CustomStack()
aligner.df = df
aligner.safes = stack.safes
aligner.orbit = stack.orbit
aligner.workdir = stack.workdir
aligner.basedir = stack.workdir
aligner.reference = stack.reference
aligner.dem_filename = dem_file
aligner.get_dem = MethodType(patched_get_dem, aligner)
aligner._get_topo_llt = MethodType(safe_get_topo_llt, aligner)
aligner._align_ref_subswath = MethodType(patched_align_ref_subswath, aligner)
aligner._align_rep_subswath = MethodType(patched_align_rep_subswath, aligner)

def silent_compute_align(self, geometry="ra", dates=None, n_jobs=1, degrees=0.01, joblib_aligning_backend="threading"):
    dates = dates or sorted(self.df.index.tolist())
    subswaths = [self.df.loc[self.reference, "subswath"]] if "subswath" in self.df.columns else [2]
    for date in [self.reference]:
        for swath in subswaths:
            self._align_ref_subswath(swath, date)
    for date in [d for d in dates if d != self.reference]:
        for swath in subswaths:
            self._align_rep_subswath(swath, date)

aligner.compute_align = MethodType(silent_compute_align, aligner)
print("✅ Aligner fully patched and ready.")

# === 🚀 Launch Coregistration
master_str = aligner.reference
slave_strs = [d for d in stack.df.index if d != master_str and d in aligner.df.index]
print(f"🎯 Master: {master_str}")
print(f"🛰️ Total slave scenes: {len(slave_strs)}")
os.chdir(workdir)

from tqdm.notebook import tqdm
for slave_str in tqdm(slave_strs, desc="🔁 Coregistering"):
    aligner.compute_align(dates=[master_str, slave_str], n_jobs=1, degrees=0.01)

print("✅ Coregistration complete.")
