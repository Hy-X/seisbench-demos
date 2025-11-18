import seisbench
import seisbench.util
from .base import BenchmarkDataset, WaveformDataWriter

from pathlib import Path
import h5py
import pandas as pd
import numpy as np


class OKLA_CLEAN(BenchmarkDataset):
    """
    OKLAHOMA dataset (removed the duplicate s and p), Hongyu Xiao

    Using the train/test split from the EQTransformer Github repository
    train/dev split defined in SeisBench
    """

    def __init__(self, **kwargs):
        citation = (
            "Hongyu Xiao Modified from STanford EArthquake Dataset (STEAD): "
            "doi:"
        )
        license = "CC BY 4.0"
        super().__init__(citation=citation, license=license, **kwargs)

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        download_instructions = (
            "Please download STEAD following the instructions at https://github.com/smousavi05/STEAD. "
            "Provide the locations of the STEAD formatted unpacked files (metadata.csv and waveform.hdf5) in the "
            "download_kwargs argument 'basepath'."
            "This step is only necessary the first time STEAD is loaded."
        )

        metadata_dict = {
            "trace_start_time": "trace_start_time",
            "trace_category": "trace_category",
            "trace_name": "trace_name",
            "p_arrival_sample": "trace_p_arrival_sample",
            "p_status": "trace_p_status",
            "p_weight": "trace_p_weight",
            "p_travel_sec": "path_p_travel_sec",
            "s_arrival_sample": "trace_s_arrival_sample",
            "s_status": "trace_s_status",
            "s_weight": "trace_s_weight",
            "s_travel_sec": "path_s_travel_sec",
            "back_azimuth_deg": "path_back_azimuth_deg",
            "snr_db": "trace_snr_db",
            "coda_end_sample": "trace_coda_end_sample",
            "network_code": "station_network_code",
            "receiver_code": "station_code",
            "receiver_type": "trace_channel",
            "receiver_latitude": "station_latitude_deg",
            "receiver_longitude": "station_longitude_deg",
            "receiver_elevation_m": "station_elevation_m",
            "source_id": "source_id",
            "source_origin_time": "source_origin_time",
            "source_origin_uncertainty_sec": "source_origin_uncertainty_sec",
            "source_latitude": "source_latitude_deg",
            "source_longitude": "source_longitude_deg",
            "source_error_sec": "source_error_sec",
            "source_gap_deg": "source_gap_deg",
            "source_horizontal_uncertainty_km": "source_horizontal_uncertainty_km",
            "source_depth_km": "source_depth_km",
            "source_depth_uncertainty_km": "source_depth_uncertainty_km",
            "source_magnitude": "source_magnitude",
            "source_magnitude_type": "source_magnitude_type",
            "source_magnitude_author": "source_magnitude_author",
        }

        path = self.path
        #basepath = '/Users/hongyuxiao/Hongyu_File/Data'
        basepath = '/ourdisk/hpc/ogs/hongyux/dont_archive/Merge_Data_OklaCLEAN_Raw'
        if basepath is None:
            raise ValueError(
                "No cached version of STEAD found. " + download_instructions
            )

        basepath = Path(basepath)

        if not (basepath / "merge.csv").is_file():
            raise ValueError(
                "Basepath does not contain file merge.csv. " + download_instructions
            )
        if not (basepath / "merge.hdf5").is_file():
            raise ValueError(
                "Basepath does not contain file merge.hdf5. " + download_instructions
            )

        self.path.mkdir(parents=True, exist_ok=True)
        seisbench.logger.warning(
            "Converting STEAD files to SeisBench format. This might take a while."
        )

        #split_url = "https://github.com/smousavi05/EQTransformer/raw/master/ModelsAndSampleData/test.npy"
        #seisbench.util.download_http(
        #    split_url, path / "test.npy", desc=f"Downloading test splits"
        #)

        # Copy metadata and rename columns to SeisBench format
        metadata = pd.read_csv(basepath / "merge.csv")
        metadata.rename(columns=metadata_dict, inplace=True)
        metadata['split'] = np.random.choice(['train', 'dev', 'test'], size=metadata.shape[0], p=[0.7, 0.15, 0.15])

	#splits = []
        #for o in metadata['source_origin_time']: 
        #    if str(o) < "2021-01-01": 
        #        split = "train" 
        #    elif str(o) < "2021-06-01": 
        #        split = "dev" 
        #    else: 
        #        split = "test" 
        #    splits.append(split) 
        #metadata["split"] = splits
        # Set split
        #test_split = set(np.load(path / "test.npy"))
        #test_mask = metadata["trace_name"].isin(test_split)
        #train_dev = metadata["trace_name"][~test_mask].values
        #dev_split = train_dev[
        #    ::18
        #]  # Use 5% of total traces as suggested in EQTransformer Github repository
        #dev_mask = metadata["trace_name"].isin(dev_split)
        #metadata["split"] = "train"
        #metadata.loc[dev_mask, "split"] = "dev"
        #metadata.loc[test_mask, "split"] = "test"

        # Writer data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 100,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        writer.set_total(len(metadata))

        with h5py.File(basepath / "merge.hdf5") as f:
            gdata = f["data"]
            for _, row in metadata.iterrows():
                row = row.to_dict()
                #print(_,row)
                #print(waveforms)
                waveforms = gdata[row["trace_name"]][()]
                #print(waveforms)
                #print(type(waveforms))
                #print(len(waveforms))
                if waveforms.shape[1] == 3:
                    waveforms = waveforms.T  # From WC to CW
                    waveforms = waveforms[[2, 1, 0]]  # From ENZ to ZNE

                    writer.add_trace(row, waveforms)
