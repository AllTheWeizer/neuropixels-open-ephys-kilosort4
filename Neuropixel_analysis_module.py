#from sympy.codegen.ast import continue_

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

def save_nam_analysis_config(
    base_path,
    record_node: str,
    experiment_label: str,
    recording_label: str,
    Neuropix_PXI_label: str,
    NIDAQ_label: str = None,
    bit_to_uv_probe_IMEC_calibration_value: float = None,
    bad_channels_list: list = None,
    block_index_val: int = None,
    configuration_file: str = "nam_analysis_config.json",
    overwrite: bool = False,
    *,
    load_adc: bool = False,
    ADC_label: str = None,
    adc_stream_name: str = None,
    adc_stream_id: str = None,
    adc_channel_ids: list = None,
    remove_first_minutes: float = None,
    adc_mode: str = None,
    adc_sources: list = None,
):
    """
    Create (or, with overwrite=True, replace) the per-recording configuration JSON.

    Probe stream: Neuropix_PXI_label (folder name under continuous/).
    Analog (ADC) stream (updated 2026-10-30), independent of the probe:
        load_adc        True to load it in cell 1.2; False = skipped.
        ADC_label       its folder name under continuous/, e.g. 'OneBox-121.OneBox-ADC' or
                        'NI-DAQmx-104.PCIe-6251'.
        adc_stream_name / adc_stream_id
                        the reader's stream identifier, resolved from the files by
                        resolve_adc_stream() (it can differ from the folder name).
        adc_channel_ids channel IDs to keep (as discovered), or None for all.
    NIDAQ_label is the older name of ADC_label and is mapped to it explicitly; giving both
    with different values is an error. The folders are not created or checked here
    (cell 0.2 checks them before calling this).

    Multi-source schema (2026-10-31, cell 0.2's ADC_MODE / ADC_SOURCES): adc_mode is one of
    'off', 'nidaq', 'onebox', 'both' and adc_sources the list made by resolve_adc_sources()
    (one dict per selected source). It replaces load_adc / ADC_label / NIDAQ_label / adc_stream_*
    / adc_channel_ids, which must then be left at their defaults.
    """
    import json
    from pathlib import Path

    for name, value in (("bit_to_uv_probe_IMEC_calibration_value", bit_to_uv_probe_IMEC_calibration_value),
                        ("bad_channels_list", bad_channels_list), ("block_index_val", block_index_val)):
        if value is None:
            raise TypeError(f"save_nam_analysis_config() needs {name}.")
    if NIDAQ_label is not None:
        if ADC_label is None:
            ADC_label = NIDAQ_label
            print(f"[save_nam_analysis_config] NIDAQ_label is the older name of ADC_label: using ADC_label = {ADC_label!r}")
        elif ADC_label != NIDAQ_label:
            raise ValueError(f"NIDAQ_label ({NIDAQ_label!r}) and ADC_label ({ADC_label!r}) differ. Give only ADC_label.")
    if load_adc and not ADC_label:
        raise ValueError("load_adc is True but ADC_label is not set.")
    if adc_mode is not None:
        if load_adc or ADC_label or adc_stream_name or adc_stream_id or adc_channel_ids is not None:
            raise ValueError("Give either adc_mode / adc_sources or the older single-ADC arguments, not both.")
        expected = selected_adc_keys(adc_mode)
        got = [src.get("key") for src in (adc_sources or [])]
        if got != expected:
            raise ValueError(f"adc_sources {got} don't match ADC_MODE {adc_mode!r} (expected {expected}). "
                             "Use nam.resolve_adc_sources().")

    base_path = Path(base_path)
    recording_dir = base_path / record_node / experiment_label / recording_label
    data_folder = recording_dir / "continuous" / Neuropix_PXI_label
    adc_folder = recording_dir / "continuous" / ADC_label if ADC_label else None
    kilosort_folder = data_folder / "kilosort4"
    config_path = recording_dir / configuration_file

    config = {
        "bit_to_uv_probe_IMEC_calibration_value": bit_to_uv_probe_IMEC_calibration_value,
        "base_path": str(base_path),
        "record_node": record_node,
        "experiment_label": experiment_label,
        "recording_label": recording_label,
        "Neuropix_PXI_label": Neuropix_PXI_label,
        "data_folder": str(data_folder),
        "kilosort_folder": str(kilosort_folder),
        "neuropix_stream_name_str": "#".join([record_node, Neuropix_PXI_label]),
        "experiment_names_str": experiment_label,
        "bad_channels_list": bad_channels_list,
        "block_index_val": block_index_val,
        "config_path": str(config_path),
        # analog (ADC) stream(s): the older single-ADC keys, replaced below by the multi-source keys
        "load_adc": bool(load_adc),
        "ADC_label": ADC_label,
        "adc_folder": str(adc_folder) if adc_folder else None,
        "adc_stream_name": adc_stream_name,
        "adc_stream_id": adc_stream_id,
        "adc_channel_ids": list(adc_channel_ids) if adc_channel_ids is not None else None,
        # requested crop, for checking saved outputs against it (None if not given)
        "remove_first_minutes": float(remove_first_minutes) if remove_first_minutes is not None else None,
    }

    if adc_mode is not None:
        for k in ("load_adc", "ADC_label", "adc_folder", "adc_stream_name", "adc_stream_id", "adc_channel_ids"):
            config.pop(k)
        config["adc_mode"] = adc_mode
        config["adc_sources"] = [dict(src) for src in (adc_sources or [])]
    if not recording_dir.is_dir():
        raise FileNotFoundError(f"Recording folder not found (configuration not written): {recording_dir}")
    existed_before = config_path.exists()
    if overwrite or not existed_before:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        action = "overwritten" if existed_before else "created"
        print(f"[save_nam_analysis_config] Configuration file {action}: {config_path}")
        return config_path

    print(f"[save_nam_analysis_config] File already exists and overwrite=False:\n  {config_path}")
    return None


def get_config_path(base_path, record_node, experiment_label, recording_label, configuration_file='nam_analysis_config.json'):
    config_save_directory = base_path / record_node / experiment_label / recording_label
    config_path = config_save_directory / configuration_file
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    return config_path

def define_analysis_paths_from_config(config_path):
    from pathlib import Path
    import json

    global si_folder, si_folder_be, si_folder_ae, si_report_folder, si_report_folder_be, si_report_folder_ae
    global phy_folder, phy_folder_be, phy_folder_ae, unit_refine_folder, preprocessed_folder
    global kilosort_sorter_output_folder, clusteringpath_folder, custom_analytics_folder, burst_type_path
    global data_folder, kilosort_folder, custom_files_folder, bombcell_folder
    global rec_step5_lfp, rec_step6_spike, rec_clean_step1, rec_clean_step2

    with open(config_path, 'r') as f:
        config = json.load(f)

    data_folder = Path(config['data_folder'])
    kilosort_folder = Path(config['kilosort_folder'])

    si_folder = data_folder / 'si_analyzer'
    si_folder_be = data_folder / 'si_analyzer_be'
    si_folder_ae = data_folder / 'si_analyzer_ae'
    si_report_folder = data_folder / 'si_report'
    si_report_folder_be = data_folder / 'si_report_be'
    si_report_folder_ae = data_folder / 'si_report_ae'
    phy_folder = data_folder / 'phy_folder'
    phy_folder_be = data_folder / 'phy_folder_be'
    phy_folder_ae = data_folder / 'phy_folder_ae'
    unit_refine_folder = data_folder / 'unit_refine'
    preprocessed_folder = data_folder / 'preprocessed_data'
    rec_step5_lfp = preprocessed_folder / 'step5_rec_lfp'
    rec_step6_spike = preprocessed_folder / 'step6_rec_spike'
    rec_clean_step1 = preprocessed_folder / 'rec_clean_step1'
    rec_clean_step2 = preprocessed_folder / 'rec_clean_step2'
    kilosort_sorter_output_folder = kilosort_folder / 'sorter_output'
    clusteringpath_folder = data_folder / 'clusteringpath'
    custom_analytics_folder = data_folder / 'custom_analytics'
    burst_type_path = custom_analytics_folder / 'burst_type'
    custom_files_folder = data_folder / 'custom_files'
    bombcell_folder = data_folder / 'bombcell'


def _adc_config_from_file(cfg):
    """
    ADC settings of a configuration file. Files written before 2026-10-30 have NI-DAQ keys
    (nidaq_folder, NIDAQ_stream_name_str) and never loaded the ADC stream: they map to
    load_adc = False and ADC_label = the NI-DAQ folder name. No stream identifier,
    calibration or other metadata is invented for them.
    """
    from pathlib import Path
    legacy_keys = ("load_adc", "ADC_label", "adc_folder", "adc_stream_name", "adc_stream_id",
                   "adc_channel_ids", "nidaq_folder", "NIDAQ_stream_name_str")
    if "adc_mode" in cfg:
        mixed = [k for k in legacy_keys if k in cfg]
        if mixed:
            raise ValueError(f"Configuration mixes ADC_MODE with older ADC keys {mixed}. Rerun cell 0.2.")
        mode = cfg["adc_mode"]
        keys = selected_adc_keys(mode)
        sources = [dict(src) for src in cfg.get("adc_sources") or []]
        if [src.get("key") for src in sources] != keys:
            raise ValueError(f"Configuration: adc_sources {[src.get('key') for src in sources]} don't match "
                             f"adc_mode {mode!r}. Rerun cell 0.2.")
        return {"adc_mode": mode, "adc_sources": sources, "load_adc": bool(sources), "ADC_label": None,
                "adc_folder": None, "adc_stream_name": None, "adc_stream_id": None, "adc_channel_ids": None,
                "adc_config_source": "ADC_MODE / ADC_SOURCES"}
    new = "ADC_label" in cfg or "load_adc" in cfg
    old = "nidaq_folder" in cfg or "NIDAQ_stream_name_str" in cfg
    if new:
        out = {
            "load_adc": bool(cfg.get("load_adc", False)),
            "ADC_label": cfg.get("ADC_label"),
            "adc_folder": Path(cfg["adc_folder"]) if cfg.get("adc_folder") else None,
            "adc_stream_name": cfg.get("adc_stream_name"),
            "adc_stream_id": cfg.get("adc_stream_id"),
            "adc_channel_ids": cfg.get("adc_channel_ids"),
            "adc_config_source": "single-ADC keys (LOAD_ADC / ADC_label, 2026-10-30)",
        }
        out["adc_mode"] = "legacy" if out["load_adc"] else "off"
        out["adc_sources"] = ([{"key": "adc", "label": out["ADC_label"],
                                "folder": str(out["adc_folder"]) if out["adc_folder"] else None,
                                "record_node": cfg.get("record_node"), "stream_name": out["adc_stream_name"],
                                "stream_id": out["adc_stream_id"], "channel_ids": out["adc_channel_ids"],
                                "unit_if_not_stated": None}] if out["load_adc"] else [])
        if old and cfg.get("nidaq_folder") and out["ADC_label"] and Path(cfg["nidaq_folder"]).name != out["ADC_label"]:
            raise ValueError(
                f"Configuration has both nidaq_folder ({cfg['nidaq_folder']}) and ADC_label ({out['ADC_label']!r}) "
                "and they disagree. Rerun cell 0.2 to rewrite the configuration."
            )
        return out
    if old:
        label = Path(cfg["nidaq_folder"]).name if cfg.get("nidaq_folder") else None
        return {
            "load_adc": False,
            "ADC_label": label,
            "adc_folder": Path(cfg["nidaq_folder"]) if cfg.get("nidaq_folder") else None,
            "adc_stream_name": None,
            "adc_stream_id": None,
            "adc_channel_ids": None,
            "adc_config_source": "older NI-DAQ keys (nidaq_folder) mapped to ADC_label; load_adc = False",
            "adc_mode": "off",
            "adc_sources": [],
        }
    return {"load_adc": False, "ADC_label": None, "adc_folder": None, "adc_stream_name": None,
            "adc_stream_id": None, "adc_channel_ids": None, "adc_config_source": "none",
            "adc_mode": "off", "adc_sources": []}


def load_experiment_config(config_path='full_analysis_config.json'):
    import json
    from pathlib import Path

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    config = {
        'bit_to_uv': cfg['bit_to_uv_probe_IMEC_calibration_value'],
        'base_path': Path(cfg['base_path']),
        'data_folder': Path(cfg['data_folder']),
        'kilosort_folder': Path(cfg['kilosort_folder']),
        'neuropix_stream_name_str': cfg['neuropix_stream_name_str'],
        'experiment_names_str': cfg['experiment_names_str'],
        'bad_channels_list': cfg['bad_channels_list'],
        'block_index_val': cfg['block_index_val'],
        "record_node": cfg["record_node"],
        "experiment_label": cfg["experiment_label"],
        "recording_label": cfg["recording_label"],
        "Neuropix_PXI_label": cfg["Neuropix_PXI_label"],
        "config_path": Path(cfg["config_path"]),
        **_adc_config_from_file(cfg),
        "remove_first_minutes": cfg.get("remove_first_minutes"),
    }
    define_analysis_paths_from_config(config_path)
    return config

import numpy as np
from pathlib import Path
from typing import Union, Optional

def load_ops(ops_path: Union[str, Path], device: Optional[str] = None):
    import numpy as np
    from pathlib import Path
    from typing import Union, Optional
    """
    Load a saved Kilosort/SpikeInterface `ops.npy` dictionary.

    If PyTorch is available, numpy arrays listed in `ops['is_tensor']`
    (and those found under `ops['preprocessing']`) are converted to
    torch tensors and moved to `device`:
        • "cuda"  → the first GPU
        • "cpu"   → main memory
        • None    → auto-select: cuda if available, else cpu

    When PyTorch is **not** installed, everything stays as NumPy arrays
    and the function still returns successfully.

    Parameters
    ----------
    ops_path : str or Path
        Path to the `ops.npy` file.
    device : {"cuda", "cpu", None}, optional
        Target device for tensors. Ignored if PyTorch is missing.

    Returns
    -------
    dict
        The `ops` dictionary (mixed tensors / NumPy arrays or only NumPy).
    """
    # ------------------------------------------------------------------ try torch
    try:
        import torch
        _has_torch = True
    except ModuleNotFoundError:
        torch, _has_torch = None, False     # keep a placeholder

    # ------------------------------------------------------------------ pick device
    if _has_torch:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)
    else:
        device = "cpu"          # force CPU when torch absent
        torch_device = None     # not used

    # ------------------------------------------------------------------ load ops
    ops_path = Path(ops_path)
    ops = np.load(ops_path, allow_pickle=True).item()

    # ------------------------------------------------------------------ convert to tensors (only if torch exists)
    if _has_torch:
        # 1) top-level keys listed in ops['is_tensor'] -------------------
        for key in ops.get("is_tensor", []):
            if key in ops and isinstance(ops[key], np.ndarray):
                ops[key] = torch.from_numpy(ops[key]).to(torch_device)

        # 2) nested preprocessing dict ----------------------------------
        if isinstance(ops.get("preprocessing"), dict):
            ops["preprocessing"] = {
                k: torch.from_numpy(v).to(torch_device) if isinstance(v, np.ndarray) else v
                for k, v in ops["preprocessing"].items()
            }

    # ------------------------------------------------------------------ done
    return ops


def inspect_and_plot_recording(rec1, si):
    import spikeinterface.full as si
    import matplotlib.pyplot as plt
    # Print out some details about the recording to verify it's loaded correctly
    print(f'Recording data: {rec1}')

    # Optionally, get the probe information if it's defined
    # This step might need adjustments based on how probe information is handled in your dataset
    if hasattr(rec1, 'get_probe'):
        df_probe = rec1.get_probe().to_dataframe()
        print(df_probe.head())

    # Plotting the probe layout if applicable and relevant
    fig, ax = plt.subplots(figsize=(15, 10))
    if hasattr(si, 'plot_probe_map'):
        si.plot_probe_map(rec1, ax=ax, with_channel_ids=False)
        ax.set_ylim(-100, 4000)  # Adjust the y-limits to fit the probe dimensions
    else:
        print("Plotting function not available.")

    # Display the figure with the probe layout
    plt.show()


def process_block(base_path, block_index, rec, event_channel="Neuropixels PXI Sync"):
    """
    Shift the recording so its first sample is t = 0 s, and shift one event channel
    by the same amount.

    event_channel : name of the Open Ephys event channel to read.
        Default "Neuropixels PXI Sync" is the probe's 1 Hz sync signal, NOT a stimulus.
        If the name is not found, the available channel names are printed in the error.

    Shifted event times are in seconds from the first sample of `rec`. If `rec` was
    cropped, events from the removed part come out negative; use
    crop_events_to_recording() to keep only events inside the recording.
    """
    import spikeinterface.full as si
    import copy

    events = si.read_openephys_event(base_path, block_index)
    available = list(events.channel_ids)
    if event_channel not in available:
        raise ValueError(
            f"Event channel '{event_channel}' not found.\n"
            f"Available event channels: {available}\n"
            "Set EVENT_CHANNEL in the configuration cell to one of these."
        )
    event_times = events.get_event_times(event_channel)

    # Start time of the (possibly cropped) recording
    start_time = rec.get_start_time()
    print(f"Block {block_index} start time:", start_time)
    print(f"Event channel used: {event_channel}  (available: {available})")

    event_times_shifted = event_times - start_time

    # Independent deep copy so shifting does not affect the original
    rec_block = copy.deepcopy(rec)
    rec_block.shift_times(-start_time)

    print("Events:", events)
    print("Raw event times:", event_times[:10])
    print("Shifted event times:", event_times_shifted[:10])

    return rec_block, events, event_times_shifted

def calculate_pulse_artifact_triggers(selected_shifted_events, selected_indexes_1, frequency, duty_cycle, pulse_train_duration, falling_edge_offset, sampling_frequency):
    import numpy as np
    total_pulse_period = 1 / frequency
    pulse_width = total_pulse_period * duty_cycle

    def generate_pulse_times(start_time, total_period, train_duration):
        times = []
        current_time = start_time
        while current_time < start_time + train_duration:
            times.append(current_time)
            current_time += total_period
        return times

    def generate_falling_times(rising_times, width, offset):
        return [time + width + offset for time in rising_times]

    selected_shifted_events_sublist = [selected_shifted_events[i] for i in selected_indexes_1 if i < len(selected_shifted_events)]

    pulse_trains_rising = {
        event_time: generate_pulse_times(event_time, total_pulse_period, pulse_train_duration)
        for event_time in selected_shifted_events_sublist
    }

    pulse_trains_falling = {
        event_time: generate_falling_times(pulse_trains_rising[event_time], pulse_width, falling_edge_offset)
        for event_time in selected_shifted_events_sublist
    }

    pulse_trains_combined = {}
    for event_time in selected_shifted_events_sublist:
        combined_times = pulse_trains_rising[event_time] + pulse_trains_falling[event_time]
        combined_times_sorted = sorted(combined_times)
        pulse_trains_combined[event_time] = combined_times_sorted

    artifact_frames_triggers = sorted(
        [int(time * sampling_frequency) for sublist in pulse_trains_combined.values() for time in sublist]
    )
    artifact_sec_triggers = np.array(
        sorted(set(float(time) for sublist in pulse_trains_combined.values() for time in sublist)), dtype="float64"
    )

    return artifact_frames_triggers, artifact_sec_triggers

def si_remove_artifacts(recording, triggers, mode, ms_before, ms_after):
    from spikeinterface.preprocessing import remove_artifacts
    recording_output = remove_artifacts(
        recording=recording,       # CORRECT param name
        list_triggers=triggers,         # CORRECT param name
        mode=mode,
        ms_before=ms_before,       # Float in milliseconds
        ms_after=ms_after          # Float in milliseconds
    )
    return recording_output

def plot_in_chunks(recording_name, recording, events, channel_ids, channels_name, start_time=0, end_time=None, chunk_duration=1000, save_folder=None):
    import spikeinterface.widgets as sw
    import matplotlib.pyplot as plt
    import numpy as np

    def get_total_duration(recording, max_duration=None):
        """
        Calculate the total duration of the recording in seconds.
        Allows limiting the duration to a maximum value.
        """
        num_frames = recording.get_num_frames()
        sampling_frequency = recording.get_sampling_frequency()
        duration = num_frames / sampling_frequency
        return min(duration, max_duration) if max_duration is not None else duration

    def load_data_chunk(recording, start_time, end_time, channel_ids):
        """
        Load a chunk of data from the recording within the specified time range.
        Only loads specified channels.
        """
        sampling_frequency = recording.get_sampling_frequency()
        start_frame = int(start_time * sampling_frequency)
        end_frame = int(end_time * sampling_frequency)
        return recording.frame_slice(start_frame, end_frame).channel_slice(channel_ids)

    if end_time is None:
        end_time = get_total_duration(recording)  # Get total duration if not provided

    chunks = np.arange(start_time, end_time, chunk_duration)
    fig, ax = plt.subplots(figsize=(15, 5))

    for start in chunks:
        actual_end = min(start + chunk_duration, end_time)
        chunk = load_data_chunk(recording, start, actual_end, channel_ids)

        if chunk is None or chunk.get_num_frames() == 0:
            print(f"No data available from {start} to {actual_end}. Skipping this chunk.")
            continue

        # Plot the chunk while adjusting time_range to match the actual global recording timeline
        sw.plot_traces(chunk, ax=ax, time_range=(start, actual_end), channel_ids=channel_ids, order_channel_by_depth=True)

        # Plot events within the global timeline
        chunk_events = [event for event in events if start <= event < actual_end]
        for event in chunk_events:
            ax.axvline(x=event, color='red', linestyle='--', label='Event' if 'Event' not in [line.get_label() for line in ax.lines] else "")

    ax.set_xlim([start_time, end_time])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Recording Data Over Time with Events')
    ax.legend()
    plt.savefig(save_folder / f'{recording_name}_{channels_name}_s{start_time}_e{end_time}.png')
    plt.show()

def save_with_spikeinterface(
    recording,
    output_folder,
    si_rec_folder='si_preprocessed',
    events=None,
    events_file_name=None,
    events_csv_file_name=None,
    overwrite=False
):
    from pathlib import Path
    import os
    import numpy as np
    import pandas as pd
    from scipy.io import savemat

    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    si_save_folder_path = output_folder / si_rec_folder

    # Check if the recording folder already exists
    if si_save_folder_path.exists() and not overwrite:
        print(f"Folder {si_save_folder_path} already exists. Data not overwritten.")
        return

    # Save recording data
    recording.save(folder=si_save_folder_path, chunks='1s', overwrite=overwrite)
    print(f"Recording data saved to {si_save_folder_path}")

    # If events filenames are provided, attempt saving events
    if events is not None and events_file_name and events_csv_file_name:
        events_file_path = output_folder / events_file_name
        events_csv_file_path = output_folder / events_csv_file_name

        if not events_file_path.exists() or overwrite:
            events_array = np.array(events)
            event_data = {'timestamps': events_array, 'peak': events_array}
            savemat(events_file_path, {'laserevents': event_data})
            print(f"Event data saved to {events_file_path}")

            event_df = pd.DataFrame(event_data)
            event_df.to_csv(events_csv_file_path, index=False)
            print(f"Event data also saved as CSV to {events_csv_file_path}")
        else:
            print(f"Event file {events_file_path} already exists. No action taken.")

def apply_unit_refinement(si_folder, unit_refine_folder):
    """
    Run the two UnitRefine models (noise/neural and SUA/MUA) on a saved
    SortingAnalyzer and save the labels with the REAL unit IDs.

    Fix: the old version numbered units 0..N-1 by position, which does not match
    Kilosort cluster IDs. Labels are now joined on the unit ID index that
    SpikeInterface returns, and checked against the analyzer's unit IDs.

    Writes unit_refine_results.csv with columns:
        unit_id, Noise_Label, SUA_MUA_Label (+ probability columns if provided)
    Returns the results DataFrame.
    """
    import spikeinterface.full as si
    import pandas as pd
    from pathlib import Path

    reloaded_analyzer = si.load_sorting_analyzer(si_folder)
    analyzer_ids = list(reloaded_analyzer.unit_ids)

    noise_neuron_labels = si.auto_label_units(
        sorting_analyzer=reloaded_analyzer,
        repo_id="SpikeInterface/UnitRefine_noise_neural_classifier",
        trust_model=True,
    )

    sua_mua_labels = si.auto_label_units(
        sorting_analyzer=reloaded_analyzer,
        repo_id="SpikeInterface/UnitRefine_sua_mua_classifier",
        trust_model=True,
    )

    def _tidy(df, prefix):
        df = pd.DataFrame(df).copy()
        if len(df) != len(analyzer_ids):
            raise ValueError(
                f"{prefix}: {len(df)} labels for {len(analyzer_ids)} units in the analyzer."
            )
        # Some SpikeInterface versions return a plain 0..N-1 index; the labels are
        # in analyzer unit order, so put the real unit IDs back. Any OTHER mismatch
        # means the labels can't be matched to units safely.
        if set(map(str, df.index)) != set(map(str, analyzer_ids)):
            positional = [str(i) for i in range(len(df))]
            if list(map(str, df.index)) != positional:
                raise ValueError(
                    f"{prefix}: returned unit IDs {list(df.index)[:5]}... match neither the analyzer's "
                    f"unit IDs {list(analyzer_ids)[:5]}... nor a plain 0..N-1 order. Labels not saved."
                )
            df.index = analyzer_ids
        df.index = pd.Index([int(u) if str(u).lstrip("-").isdigit() else u for u in df.index],
                            name="unit_id")
        out = pd.DataFrame(index=df.index)
        out[f"{prefix}_Label"] = df["prediction"]
        if "probability" in df.columns:
            out[f"{prefix}_Probability"] = df["probability"]
        return out

    noise_df = _tidy(noise_neuron_labels, "Noise")
    sua_df = _tidy(sua_mua_labels, "SUA_MUA")

    results_df = noise_df.join(sua_df, how="outer").reset_index()

    unit_refine_folder = Path(unit_refine_folder)
    unit_refine_folder.mkdir(parents=True, exist_ok=True)
    csv_path = unit_refine_folder / "unit_refine_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(results_df)
    print(f"Results saved to {csv_path}")
    return results_df

def combine_cluster_categorizations(unit_refine_folder, kilosort_sorter_output_folder, data_folder):
    import os
    import pandas as pd

    custom_analytics_folder = data_folder / 'custom_analytics'
    unit_refine_path = os.path.join(unit_refine_folder, 'unit_refine_results.csv')
    ks_label_path = os.path.join(kilosort_sorter_output_folder, 'cluster_KSLabel.tsv')
    bc_unit_type_path = os.path.join(kilosort_sorter_output_folder, 'cluster_bc_unitType.tsv')

    # Load datasets
    unit_refine_results = pd.read_csv(unit_refine_path)
    ks_labels = pd.read_csv(ks_label_path, sep='\t')
    bc_unit_type = pd.read_csv(bc_unit_type_path, sep='\t')

    # Rename columns for clarity
    unit_refine_results.rename(columns={
        'Noise_Label': 'Noise_Label (unit_refine)',
        'SUA_MUA_Label': 'SUA_MUA_Label (unit_refine)',
        'Unit_ID': 'cluster_id',
        'unit_id': 'cluster_id',
    }, inplace=True)
    ks_labels.rename(columns={'KSLabel': 'KSLabel (kilosort4)'}, inplace=True)
    bc_unit_type.rename(columns={'bc_unitType': 'bc_unitType (bombcell)'}, inplace=True)

    # Merge datasets
    merged_data = pd.merge(unit_refine_results, ks_labels, on='cluster_id')
    final_merged_data = pd.merge(merged_data, bc_unit_type, on='cluster_id')

    # Save merged results
    output_path = os.path.join(custom_analytics_folder, 'kilo-bombcell-unitrefine_clusterID_categorization.csv')
    try:
        final_merged_data.to_csv(output_path, index=False)
        print(f"Combined data saved to {output_path}")
    except PermissionError:
        print(
            f"Permission denied: Cannot write to {output_path}. Please close the file if it is open elsewhere or check file permissions.")

    # Extract and print the requested lists
    sua_units = final_merged_data[final_merged_data['SUA_MUA_Label (unit_refine)'] == 'sua']['cluster_id'].tolist()
    good_ks_units = final_merged_data[final_merged_data['KSLabel (kilosort4)'] == 'good']['cluster_id'].tolist()
    good_bc_units = final_merged_data[final_merged_data['bc_unitType (bombcell)'] == 'GOOD']['cluster_id'].tolist()

    # Units common to all three lists
    common_good_sua_units = list(set(sua_units) & set(good_ks_units) & set(good_bc_units))

    # Printing the lists
    print("\nUnits labeled as 'SUA' by UnitRefine:", sua_units)
    print("Units labeled as 'good' by Kilosort:", good_ks_units)
    print("Units labeled as 'GOOD' by Bombcell:", good_bc_units)
    print("Units labeled as 'SUA', 'good', and 'GOOD':", common_good_sua_units)

    # Save lists as separate files
    pd.Series(sua_units).to_csv(os.path.join(custom_analytics_folder, 'sua_units.csv'), index=False, header=False)
    pd.Series(good_ks_units).to_csv(os.path.join(custom_analytics_folder, 'good_ks_units.csv'), index=False, header=False)
    pd.Series(good_bc_units).to_csv(os.path.join(custom_analytics_folder, 'good_bc_units.csv'), index=False, header=False)
    pd.Series(common_good_sua_units).to_csv(os.path.join(custom_analytics_folder, 'common_good_sua_units.csv'), index=False, header=False)

    return sua_units, good_ks_units, good_bc_units, common_good_sua_units

def copy_kilosort_files(kilosort_sorting_output_folder, clusteringpath_folder):

    import shutil
    # Check if the destination folder exists, and create it if it doesn't
    if not clusteringpath_folder.exists():
        clusteringpath_folder.mkdir(parents=True, exist_ok=True)
        print(f"Folder '{clusteringpath_folder}' created.")
    else:
        print(f"Folder '{clusteringpath_folder}' already exists.")

    # Copy contents from kilosort_sorting_output_folder to clusteringpath_folder
    for item in kilosort_sorting_output_folder.iterdir():
        dest = clusteringpath_folder / item.name
        if item.is_dir():
            # Copy directory recursively
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            # Copy file
            shutil.copy2(item, dest)

    print("Contents copied successfully from kilosort sorting output folder to clustering path folder.")


def classify_cell_types(kilosort_folder, custom_analytics_folder, selected_units, filename):
    import pandas as pd
    from scipy.io import loadmat

    cell_metrics_path = kilosort_folder / 'traces_cached_seg0.cell_metrics.cellinfo.mat'

    if not cell_metrics_path.exists():
        raise FileNotFoundError(f"No file found at {cell_metrics_path}")

    cell_metrics = loadmat(cell_metrics_path, simplify_cells=True)['cell_metrics']

    # Get MATLAB cell metrics data
    matlab_cluster_ids = cell_metrics['cluID']
    matlab_cell_types = cell_metrics['putativeCellType']

    # Create DataFrame to map unit IDs to their types
    matlab_df = pd.DataFrame({
        'cluster_id': matlab_cluster_ids,
        'putativeCellType': matlab_cell_types
    })

    # Filter to keep only good_units
    classification_df = matlab_df[matlab_df['cluster_id'].isin(selected_units)]

    # Save classification if it doesn't already exist
    if filename is not None:
        custom_good_units_classification_path = custom_analytics_folder / filename
        classification_df.to_csv(custom_good_units_classification_path, sep='\t', index=False)
        print(f"Classification saved to: {custom_good_units_classification_path}")
    else:
        print("Classification not saved because filename is None.")

    # Extract cluster IDs by cell type
    pyramidal_clusters = classification_df[classification_df['putativeCellType'] == 'Pyramidal Cell'][
        'cluster_id'].tolist()
    narrow_inhibitory_clusters = classification_df[classification_df['putativeCellType'] == 'Narrow Interneuron'][
        'cluster_id'].tolist()
    wide_inhibitory_clusters = classification_df[classification_df['putativeCellType'] == 'Wide Interneuron'][
        'cluster_id'].tolist()

    print(f"Pyramidal Clusters: {pyramidal_clusters}")
    print(f"Narrow Inhibitory Clusters: {narrow_inhibitory_clusters}")
    print(f"Wide Inhibitory Clusters: {wide_inhibitory_clusters}")

    return pyramidal_clusters, narrow_inhibitory_clusters, wide_inhibitory_clusters

def analyze_spike_amplitudes(
    selected_event_times,
    selected_event_indices,
    spike_times,
    spike_clusters,
    amplitudes,
    ops,
    cluster_ids_to_plot,
    window_seconds,
    save_path,
    opto_spk_amps,
    laser_distance,
    table_path,
    overall_table_path,
    overwrite=False
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu, shapiro, ttest_rel, wilcoxon
    from pathlib import Path

    log_msgs = []

    def _maybe_write(obj, dst: Path, *, overwrite: bool, **kw):
        if dst.exists() and not overwrite:
            print(f"Skipped (exists, overwrite=False): {dst}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(obj, "to_csv"):
            obj.to_csv(dst, **kw)
        elif hasattr(obj, "savefig"):
            obj.savefig(dst, **kw)
        else:
            dst.write_text(str(obj))
        log_msgs.append(f"{'Over-written' if overwrite else 'Saved'} → {dst}")

    def sem(series):
        return series.std() / np.sqrt(len(series))

    def test_significance(group):
        before = group['Avg_Before_Amplitude'].dropna()
        after = group['Avg_After_Amplitude'].dropna()

        # must have the same length after dropping NaNs
        n = min(len(before), len(after))
        if n < 3:
            return {'Significance': 'Insufficient Data', 'P_Value': None}

        # align and compute differences
        diffs = after.iloc[:n].values - before.iloc[:n].values

        # 1. test normality of differences
        sw_stat, sw_p = shapiro(diffs)
        normal = (sw_p > 0.05)

        # 2. choose test
        if normal:
            stat, p_value = ttest_rel(before.iloc[:n], after.iloc[:n])
        else:
            stat, p_value = wilcoxon(before.iloc[:n], after.iloc[:n])

        # 3. label effect
        if p_value < 0.05:
            result = 'Increase' if diffs.mean() > 0 else 'Decrease'
        else:
            result = 'No Change'

        return {
            'Test': 'paired t-test' if normal else 'wilcoxon signed-rank',
            'Normality_p': sw_p,
            'Significance': result,
            'P_Value': p_value
        }

    filtered_event_times = [selected_event_times[i] for i in selected_event_indices]

    # Convert spike_times from samples to seconds
    filtered_spike_times = {cluster_id: spike_times[spike_clusters == cluster_id] / ops['fs'] for cluster_id in cluster_ids_to_plot}
    filtered_amplitudes = {cluster_id: amplitudes[spike_clusters == cluster_id] for cluster_id in cluster_ids_to_plot}

    window_duration = 2 * window_seconds  # Duration of the window in seconds

    # Analyzing each cluster separately and collectively
    all_relative_spike_times = []
    all_spike_amplitudes = []

    analytics_df = pd.DataFrame({
        'Cluster_ID': pd.Series(dtype='int'),
        'Event_Index': pd.Series(dtype='int'),
        'Event_Time': pd.Series(dtype='float'),
        'Total_Event_Spikes': pd.Series(dtype='int'),
        'Avg_Amplitude': pd.Series(dtype='float'),
        'Avg_Before_Amplitude': pd.Series(dtype='float'),
        'Avg_After_Amplitude': pd.Series(dtype='float')
    })

    # Correct the part of your code where you collect data for analytics
    analytics_data = []
    for cluster_id in cluster_ids_to_plot:
        # Initialize individual lists inside the loop for each cluster
        individual_relative_spike_times = []
        individual_after_relative_spike_times = []
        individual_spike_amplitudes = []
        individual_before_amplitudes = []
        individual_after_amplitudes = []

        for event_idx, event_time_sec in enumerate(filtered_event_times):
            start_window = event_time_sec - window_seconds
            end_window = event_time_sec + window_seconds

            # Filter spikes within this window
            event_indices = (filtered_spike_times[cluster_id] >= start_window) & (filtered_spike_times[cluster_id] <= end_window)
            event_spike_times = filtered_spike_times[cluster_id][event_indices]
            event_spike_amplitudes = filtered_amplitudes[cluster_id][event_indices]
            relative_spike_times = event_spike_times - event_time_sec
            after_relative_spike_times_mask = (relative_spike_times >= 0)
            after_relative_spike_times = relative_spike_times[after_relative_spike_times_mask]
            before_amplitudes = event_spike_amplitudes[relative_spike_times < 0]
            after_amplitudes = event_spike_amplitudes[relative_spike_times >= 0]
            individual_before_amplitudes.extend(before_amplitudes)
            individual_after_amplitudes.extend(after_amplitudes)
            individual_relative_spike_times.extend(relative_spike_times)  # Extend the individual times
            individual_spike_amplitudes.extend(event_spike_amplitudes)    # Extend the individual amplitudes
            all_relative_spike_times.extend(relative_spike_times)
            all_spike_amplitudes.extend(event_spike_amplitudes)

            # Append analytics data for each event
            analytics_data.append({
                'Cluster_ID': cluster_id,
                'Event_Index': event_idx,
                'Event_Time': event_time_sec,
                'Total_Event_Spikes': len(event_spike_times),
                'Avg_Amplitude': np.mean(event_spike_amplitudes) if len(event_spike_amplitudes) > 0 else np.nan,
                'Avg_Before_Amplitude': np.mean(before_amplitudes) if len(before_amplitudes) > 0 else np.nan,
                'Avg_After_Amplitude': np.mean(after_amplitudes) if len(after_amplitudes) > 0 else np.nan
            })

        temp_df = pd.DataFrame(analytics_data)
        temp_df = temp_df.dropna(axis=1, how='all')
        analytics_df = pd.concat([analytics_df, temp_df], ignore_index=True)
        cluster_data = analytics_df[analytics_df['Cluster_ID'] == cluster_id]
        mean_before = cluster_data['Avg_Before_Amplitude'].mean() if not cluster_data['Avg_Before_Amplitude'].isna().all() else 0
        sem_before = sem(cluster_data['Avg_Before_Amplitude'].dropna()) if not cluster_data['Avg_Before_Amplitude'].dropna().empty else 0
        mean_after = cluster_data['Avg_After_Amplitude'].mean() if not cluster_data['Avg_After_Amplitude'].isna().all() else 0
        sem_after = sem(cluster_data['Avg_After_Amplitude'].dropna()) if not cluster_data['Avg_After_Amplitude'].dropna().empty else 0
        analytics_data = [] # Clear analytics_data for the next cluster

        # Optionally, plot individual cluster analysis
        amp_path = opto_spk_amps / f'Spike_Amplitude_Analytics_{laser_distance}mm_{cluster_id}.png'
        plt.figure(figsize=(10, 5))
        fig = plt.gcf()
        plt.scatter(individual_relative_spike_times, individual_spike_amplitudes, alpha=0.5)
        plt.axvline(x=0, color='red', linestyle='--', label='Event Time')
        plt.xlabel('Time Relative to Event (seconds)')
        plt.ylabel('Spike Amplitude')
        plt.title(f'Spike Times and Amplitudes - Cluster {cluster_id} ({laser_distance}mm)')
        plt.xlim([-window_seconds, window_seconds])
        plt.grid(True)
        plt.legend()
        _maybe_write(fig, amp_path, overwrite=overwrite, dpi=300)
        plt.close(fig)

        # Plot bar graph comparing before and after event spike amplitudes
        fig, ax = plt.subplots(figsize=(8, 6))
        fig_bar = plt.gcf()
        # Set the bar positions on the x-axis
        x_positions = [0, 1]  # Positions for 'Before' and 'After'
        print(f"Mean Before: {mean_before}, SEM Before: {sem_before}")
        print(f"Mean After: {mean_after}, SEM After: {sem_after}")

        means = [mean_before, mean_after]
        sems = [sem_before, sem_after]

        bars = ax.bar(['Before', 'After'], [mean_before, mean_after], yerr=[sem_before, sem_after], capsize=5, color=['blue', 'orange'])
        ax.set_ylabel('Average Spike Amplitude')
        ax.set_title(f'Average Spike Amplitude Before and After Event - Cluster {cluster_id}')
        #ax.set_ylim(0, max_amplitude)  # Use the maximum amplitude for all plots
        plt.grid(True)
        bar_path = opto_spk_amps / f'Bar_Graph_Amplitude_{laser_distance}mm_{cluster_id}.png'
        _maybe_write(fig_bar, bar_path, overwrite=overwrite, dpi=300)
        plt.close(fig_bar)

    # Calculate overall analytics per cluster
    overall_analytics = analytics_df.groupby('Cluster_ID').agg({
        'Event_Index': [('Events_Analyzed', 'count')],
        'Total_Event_Spikes': [('Total_Events_Spikes', 'sum')],
        'Avg_Amplitude': [('_mean', 'mean'),  # Mean spike amplitude
                                      ('_SEM', sem)],     # SEM of spike amplitude
        'Avg_Before_Amplitude': [('_mean', 'mean'),  # Mean spike amplitude
                                      ('_SEM', sem)],     # SEM of spike amplitude
        'Avg_After_Amplitude': [('_mean', 'mean'),  # Mean spike amplitude
                                      ('_SEM', sem)]     # SEM of spike amplitude
    }).reset_index()

    overall_analytics.columns = ['_'.join(col) if type(col) is tuple else col for col in overall_analytics.columns]

    # Flatten the MultiIndex in columns correctly
    overall_analytics.columns = [
        'Cluster_ID',
        'Events_Analyzed',
        'Total_Events_Spikes',
        'Avg_Events_Spike_Amplitude',
        'SEM_Events_Spike_Amplitude',
        'Avg_Before_Events_Spike_Amplitude',
        'SEM_Before_Events_Spike_Amplitude',
        'Avg_After_Events_Spike_Amplitude',
        'SEM_After_Events_Spike_Amplitude'
    ]

    # Assuming 'test_significance' returns {'Significance': result, 'P_Value': p_value}
    amplitude_data = analytics_df.groupby('Cluster_ID').apply(test_significance, include_groups=False).reset_index()

    amplitude_data = (
        analytics_df
          .groupby('Cluster_ID')
          .apply(test_significance, include_groups=False)   # returns a Series/dict with 'Significance','P_Value', etc.
          .reset_index()
    )

    # If amplitude_data has a column resulting from apply that contains dictionaries:
    if 0 in amplitude_data.columns:  # Check if this column actually exists
        amplitude_data = pd.concat([
            amplitude_data.drop(columns=[0]),  # Drop the column containing dictionaries
            amplitude_data[0].apply(pd.Series)  # Convert each dictionary in the column into a DataFrame row
        ], axis=1)
    else:
        print("Column '[0]' does not exist. Check the structure of 'amplitude_data'.")

    # Rename the columns after ensuring they exist
    #if 'Significance' in amplitude_data and 'P_Value' in amplitude_data:
        #amplitude_data.columns = ['Cluster_ID', 'Significance', 'P_Value']

    # Merge this with the existing overall_analytics DataFrame
    overall_analytics = pd.merge(overall_analytics, amplitude_data, on='Cluster_ID', how='left')

    # Save analytics data
    _maybe_write(analytics_df, table_path, overwrite=overwrite, index=False)
    _maybe_write(overall_analytics, overall_table_path, overwrite=overwrite, index=False)

    # Display analytics table
    print("Individual Event Analytics:")
    print(analytics_df)
    print("\nOverall Cluster Analytics:")
    print(overall_analytics)

    return log_msgs

def run_first_spike_jitter_analysis(
        clusters_of_interest, spike_clusters, spike_times_sec,
        amplitudes, event_times, n_periods, pulse_period,
        window_seconds, custom_analytics_folder,              # <- unchanged
        overwrite=False                                       # <- NEW ARG
):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.ticker as ticker

    # ──────────────────────────────────────────────────────────────
    # helper: save *obj* unless file exists & overwrite=False
    # ──────────────────────────────────────────────────────────────
    def _maybe_write(obj, dst: Path, *, overwrite: bool, **kw):
        """
        Save *obj* to *dst* while honouring the global `overwrite` flag.

        • Prints “Saved → …”, “Over-written → …” or
          “Skipped (exists, overwrite=False): …” accordingly.
        • Works with objects exposing .to_csv() / .savefig(), or any stringable
          object (fallback → raw text).
        """
        if dst.exists() and not overwrite:
            print(f"Skipped (exists, overwrite=False): {dst}")
            return                                          # nothing written

        dst.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(obj, "to_csv"):
            obj.to_csv(dst, **kw)
        elif hasattr(obj, "savefig"):
            obj.savefig(dst, **kw)
        else:
            dst.write_text(str(obj))

        verb = "Over-written" if dst.exists() and overwrite else "Saved"
        print(f"{verb} → {dst}")

    # ──────────────────────────────────────────────────────────────
    def idx_to_time(x): return x * pulse_period
    def time_to_idx(t): return t / pulse_period
    # ──────────────────────────────────────────────────────────────

    for cluster_id in clusters_of_interest:
        plt.clf()
        clus_idx   = np.where(spike_clusters == cluster_id)[0]
        times      = spike_times_sec[clus_idx]
        records    = []

        # ------------------------------------------------- gather data
        for evt_i, t_event in enumerate(event_times):
            for p_i in range(n_periods):
                t0   = t_event + p_i * pulse_period
                mask = (times >= t0) & (times < t0 + pulse_period)
                if not np.any(mask):
                    continue
                rel   = np.where(mask)[0][0]       # first in window
                jit   = (times[rel] - t0) * 1e3    # ms
                amp   = amplitudes[clus_idx[rel]]
                records.append(dict(
                    Cluster_ID   = cluster_id,
                    Event_Index  = evt_i,
                    Pulse_Index  = p_i,
                    Jitter_ms    = jit,
                    Amplitude    = amp
                ))

        df = pd.DataFrame(records)
        if df.empty:
            print(f"Cluster {cluster_id}: no first-spike data in 0–{window_seconds}s window.")
            continue

        # ------------------------------------------------- raw table
        opto_1st_spk_hist = Path(custom_analytics_folder / "opto_1st_spk_hist")
        opto_1st_spk_hist.mkdir(exist_ok=True)
        csv_jitter = opto_1st_spk_hist / f"FirstSpikeJitter_allPulses_C{cluster_id}.csv"
        _maybe_write(df, csv_jitter, overwrite=overwrite, index=False)

        # ------------------------------------------------- histogram
        plt.figure(figsize=(6, 4))
        plt.hist(df["Jitter_ms"], bins=25, edgecolor="k")
        plt.xlabel("First-spike jitter (ms)")
        plt.ylabel("Count")
        plt.title(f"Cluster {cluster_id}: First-Spike Jitter\n(all pulses & events)")
        plt.grid(True); plt.tight_layout()
        png_hist = opto_1st_spk_hist / f"FirstSpikeJitterHist_allPulses_C{cluster_id}.png"
        _maybe_write(plt.gcf(), png_hist, overwrite=overwrite, dpi=300)
        plt.close()

        # ------------------------------------------------- per-pulse stats
        jit_sum = (df.groupby("Pulse_Index")["Jitter_ms"]
                     .agg(["mean", "std", "count"])
                     .reset_index()
                     .rename(columns=dict(mean="Mean_Jitter_ms",
                                          std="STD_Jitter_ms",
                                          count="Num_Spikes")))
        jit_sum["Total_Events"]       = len(event_times)
        jit_sum["Pct_Events_Spiked"]  = jit_sum["Num_Spikes"] / len(event_times) * 100

        jit_file_dir = Path(custom_analytics_folder / "opto_1st_spk_jit");
        jit_file_dir.mkdir(exist_ok=True)
        csv_sum = jit_file_dir / f"FirstSpikeJitter_SummaryByPulse_C{cluster_id}.csv"
        _maybe_write(jit_sum, csv_sum, overwrite=overwrite, index=False)

        # ------------------------------------------------- mean-jitter plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(jit_sum["Pulse_Index"], jit_sum["Mean_Jitter_ms"],
                    yerr=jit_sum["STD_Jitter_ms"], marker="o", capsize=3,
                    label="Mean ± 1 σ")
        ax.legend(); ax.grid(True); plt.tight_layout()
        ax.set_xlabel(f"Pulse Index (0→{n_periods})")
        ax.set_ylabel("Mean First-Spike Jitter (ms)")
        ax.set_title(f"Cluster {cluster_id}: Mean Jitter per Pulse\n(up to {window_seconds}s)")
        sec = ax.secondary_xaxis("top", functions=(idx_to_time, time_to_idx))
        sec.set_xlabel("Time after event (s)")
        sec.set_xticks(np.arange(0, window_seconds + pulse_period/2, pulse_period*5))

        png_mean = jit_file_dir / f"FirstSpikeJitter_MeanByPulse_C{cluster_id}.png"
        _maybe_write(fig, png_mean, overwrite=overwrite, dpi=300)
        plt.close()

        # ------------------------------------------------- amplitude summary
        amp_sum = (df.groupby("Pulse_Index")["Amplitude"]
                     .agg(["mean", "sem", "count"])
                     .reset_index()
                     .rename(columns=dict(mean="Mean_Amplitude",
                                          sem="SEM_Amplitude",
                                          count="Num_Spikes")))
        amp_sum["Total_Events"]      = len(event_times)
        amp_sum["Pct_Events_Spiked"] = amp_sum["Num_Spikes"] / len(event_times) * 100

        amp_file_dir = Path(custom_analytics_folder / "opto_1st_spk_amp");
        amp_file_dir.mkdir(exist_ok=True)
        csv_amp = amp_file_dir / f"FirstSpikeAmplitude_SummaryByPulse_C{cluster_id}.csv"
        _maybe_write(amp_sum, csv_amp, overwrite=overwrite, index=False)

        # ------------------------------------------------- mean-amplitude plot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.errorbar(amp_sum["Pulse_Index"], amp_sum["Mean_Amplitude"],
                     yerr=amp_sum["SEM_Amplitude"], marker="o", capsize=3,
                     label="Mean Amp ± 1 SEM")
        ax2.legend(); ax2.grid(True); plt.tight_layout()
        ax2.set_xlabel("Pulse Index")
        ax2.set_ylabel("Mean First-Spike Amplitude")
        ax2.set_title(f"Cluster {cluster_id}: Mean First-Spike Amplitude per Pulse\n(up to {window_seconds}s)")
        sec2 = ax2.secondary_xaxis("top", functions=(idx_to_time, time_to_idx))
        sec2.set_xlabel("Time after event (s)")
        sec2.set_xticks(np.arange(0, window_seconds + pulse_period/2, pulse_period*5))

        png_amp = amp_file_dir / f"FirstSpikeAmplitude_MeanByPulse_C{cluster_id}.png"
        _maybe_write(fig2, png_amp, overwrite=overwrite, dpi=300)
        plt.close()

##############################################################################
# Sliding-window PSTH / spike-rate analytics
# • Pass overwrite=True to replace existing files
# • Optionally supply table_path / overall_table_path to let the
#   function write the CSVs for you (they are still returned, so the
#   caller can decide whether to save them elsewhere).
##############################################################################

def analyze_event_spike_rates(
        selected_shifted_events,
        selected_event_indices,
        clusters_of_interest,
        spike_clusters,
        spike_times_sec,
        window_seconds,
        baseline_start,
        baseline_end,
        window_width,
        step_size,
        window_half,
        bin_size,
        laser_distance,
        save_path,
        overwrite=False,                      # ← NEW
        table_path=None,                      # ← NEW (optional)
        overall_table_path=None               # ← NEW (optional)
):
    # ────────────────────────── imports & helper ────────────────────────
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import shapiro, ttest_rel, wilcoxon
    from pathlib import Path

    # ---- overwrite-aware writer ---------------------------------------
    def _maybe_write(obj, dst: Path, *, overwrite: bool, **kw):
        exists_before = dst.exists()  # ← NEW
        if exists_before and not overwrite:
            print(f"Skipped (exists, overwrite=False): {dst}")
            return

        dst.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(obj, "savefig"):
            obj.savefig(dst, **kw)
        elif hasattr(obj, "to_csv"):
            obj.to_csv(dst, **kw)
        else:
            dst.write_text(str(obj))

        verb = "Over-written" if exists_before and overwrite else "Saved"
        print(f"{verb} → {dst}")

    # ---- misc helpers --------------------------------------------------
    sem = lambda s: s.std() / np.sqrt(s.count()) if s.count() > 1 else np.nan   # count() ignores NaN

    def paired_change(group):
        """Return a Series with test result & p-value for before vs after."""
        before = group['Avg_Before_Event_Spike_Rate (spikes/s)'].dropna()
        after  = group['Avg_After_Event_Spike_Rate (spikes/s)'].dropna()
        n      = min(len(before), len(after))
        if n < 3:
            return pd.Series({'Test': None,
                              'Normality_p': None,
                              'Significance': 'Insufficient Data',
                              'P_Value': None})

        before = before.iloc[:n]
        after  = after.iloc[:n]
        diffs  = after.values - before.values

        sw_stat, sw_p = shapiro(diffs)
        normal = sw_p > 0.05
        if normal:
            stat, p = ttest_rel(before, after)
            tname   = 'paired t-test'
        else:
            stat, p = wilcoxon(before, after)
            tname   = 'Wilcoxon signed-rank'

        direction = ('Increase' if diffs.mean() > 0 else 'Decrease') if p < 0.05 else 'No Change'
        return pd.Series({'Test': tname,
                          'Normality_p': sw_p,
                          'Significance': direction,
                          'P_Value': p})

    # ────────────────────────── analysis body ───────────────────────────
    filtered_event_times = [selected_shifted_events[i] for i in selected_event_indices]
    analytics_rows   = []
    baseline_rates   = {}
    sliding_bins     = np.arange(-window_seconds + step_size,
                                 window_seconds,
                                 step_size)
    sliding_bins     = sliding_bins[~np.isclose(sliding_bins, 0.0, atol=1e-12)]

    # ---- compute per-event baseline rates ------------------------------
    for cid in clusters_of_interest:
        mask   = spike_clusters == cid
        spikes = spike_times_sec[mask]
        brates = {}
        for ev_t in filtered_event_times:
            b_start   = ev_t + baseline_start
            b_end     = ev_t + baseline_end
            n_spikes  = np.logical_and(spikes >= b_start, spikes < b_end).sum()
            b_period  = b_end - b_start
            # A zero baseline is a real rate of 0; the ratio to it is undefined (NaN), not huge
            brates[ev_t] = n_spikes / b_period if b_period > 0 else np.nan
        baseline_rates[cid] = brates

    # ---- main loop: cluster × event ------------------------------------
    for cid in clusters_of_interest:
        mask   = spike_clusters == cid
        spikes = spike_times_sec[mask]

        rate_lists, norm_rate_lists, bin_centers_lists = [], [], []

        plt.figure(figsize=(10, 5))

        for ev_idx, ev_t in enumerate(filtered_event_times):
            rel_times       = spikes - ev_t
            total_in_window = ((spikes >= ev_t - window_seconds) &
                               (spikes <= ev_t + window_seconds)).sum()

            rates, centers = [], []
            for bin_c in sliding_bins:
                w_start = bin_c - window_half
                w_end   = bin_c + window_half
                counts, _ = np.histogram(rel_times, bins=[w_start, w_end])
                rates.append(counts[0] / window_width)
                centers.append(bin_c)

            rates   = np.asarray(rates)
            centers = np.asarray(centers)

            # ---- normalise by baseline ---------------------------------
            base_rate   = baseline_rates[cid][ev_t]
            if np.isfinite(base_rate) and base_rate > 0:
                norm_rates = rates / base_rate
            else:
                norm_rates = np.full_like(rates, np.nan, dtype=float)   # undefined: no baseline spikes

            rate_lists.append(rates)
            norm_rate_lists.append(norm_rates)
            bin_centers_lists.append(centers)

            # scatter for individual events
            jitter = np.random.normal(0, bin_size * 0.05, size=len(centers))
            plt.scatter(centers + jitter, rates, alpha=0.8, edgecolor='black',
                        label=f'Event {ev_idx+1}' if ev_idx == 0 else "", zorder=2)

            # save per-event analytics row
            analytics_rows.append({
                'Cluster_ID'                                : cid,
                'Event_Index'                               : ev_idx,
                'Event_Time'                                : ev_t,
                'Total_Event_Spikes'                        : total_in_window,
                'Baseline_Rate (spikes/s)'                  : base_rate,
                'Avg_Event_Spike_Rate (spikes/s)'           : rates.mean(),
                'Avg_Before_Event_Spike_Rate (spikes/s)'    : rates[centers <  0].mean() if (centers <  0).any() else np.nan,
                'Avg_After_Event_Spike_Rate (spikes/s)'     : rates[centers >  0].mean() if (centers >  0).any() else np.nan,
                'Norm_Avg_Event_Spike_Rate (spikes/s)'      : norm_rates.mean(),
                'Norm_Avg_Before_Event_Spike_Rate (spikes/s)': norm_rates[centers < 0].mean() if (centers < 0).any() else np.nan,
                'Norm_Avg_After_Event_Spike_Rate (spikes/s)' : norm_rates[centers > 0].mean() if (centers > 0).any() else np.nan
            })

        # ---- cluster-level PSTH plot -----------------------------------
        rates_arr = np.vstack(rate_lists)
        mean_r    = rates_arr.mean(axis=0)
        sem_r     = rates_arr.std(axis=0, ddof=1) / np.sqrt(rates_arr.shape[0])

        centers   = bin_centers_lists[0]   # identical for all events
        plt.bar(centers, mean_r, width=bin_size*0.9, color='blue',
                yerr=sem_r, capsize=3, label='Mean Rate ± SEM', zorder=1)
        plt.axvline(0, color='blue', ls='--', label='Event Time')
        plt.xlabel('Time Relative to Event (s)')
        plt.ylabel('Average Spike Rate (spikes/s)')
        plt.title(f'PSTH – Cluster {cid} ({laser_distance} mm)')
        plt.grid(True); plt.legend()

        fig = plt.gcf()
        png_path = save_path / f'Spike_Rate_Analytics_{laser_distance}mm_{cid}.png'
        _maybe_write(fig, png_path, overwrite=overwrite, dpi=300)
        plt.close(fig)

    # ---- convert to DataFrame & summary tables ------------------------
    analytics_df = pd.DataFrame(analytics_rows)

    overall_analytics = (
        analytics_df
          .groupby('Cluster_ID')
          .agg(
              Events_Analyzed  = ('Event_Index', 'count'),
              Total_Events_Spikes = ('Total_Event_Spikes', 'sum'),
              Avg_Event_Spike_Rate_Mean        = ('Avg_Event_Spike_Rate (spikes/s)', 'mean'),
              Avg_Event_Spike_Rate_SEM         = ('Avg_Event_Spike_Rate (spikes/s)', sem),
              Avg_Before_Event_Spike_Rate_Mean = ('Avg_Before_Event_Spike_Rate (spikes/s)', 'mean'),
              Avg_Before_Event_Spike_Rate_SEM  = ('Avg_Before_Event_Spike_Rate (spikes/s)',  sem),
              Avg_After_Event_Spike_Rate_Mean  = ('Avg_After_Event_Spike_Rate (spikes/s)',  'mean'),
              Avg_After_Event_Spike_Rate_SEM   = ('Avg_After_Event_Spike_Rate (spikes/s)',   sem),
              Norm_Avg_Event_Spike_Rate_Mean        = ('Norm_Avg_Event_Spike_Rate (spikes/s)', 'mean'),
              Norm_Avg_Event_Spike_Rate_SEM         = ('Norm_Avg_Event_Spike_Rate (spikes/s)', sem),
              Norm_Avg_Before_Event_Spike_Rate_Mean = ('Norm_Avg_Before_Event_Spike_Rate (spikes/s)', 'mean'),
              Norm_Avg_Before_Event_Spike_Rate_SEM  = ('Norm_Avg_Before_Event_Spike_Rate (spikes/s)',  sem),
              Norm_Avg_After_Event_Spike_Rate_Mean  = ('Norm_Avg_After_Event_Spike_Rate (spikes/s)',   'mean'),
              Norm_Avg_After_Event_Spike_Rate_SEM   = ('Norm_Avg_After_Event_Spike_Rate (spikes/s)',    sem),
              Baseline_Rate_Mean                    = ('Baseline_Rate (spikes/s)', 'mean'),
              Baseline_Rate_SEM                     = ('Baseline_Rate (spikes/s)',  sem),
              Norm_Events_Undefined                 = ('Norm_Avg_Event_Spike_Rate (spikes/s)',
                                                       lambda s: int(s.isna().sum()))   # zero-baseline events
          )
          .reset_index()
    )

    # add significance column
    sig_df = (
        analytics_df
          .groupby('Cluster_ID')
          .apply(paired_change, include_groups=False)
          .reset_index()
    )
    overall_analytics = overall_analytics.merge(sig_df, on='Cluster_ID', how='left')

    # ---- optional CSV writing ----------------------------------------
    if table_path is not None:
        _maybe_write(analytics_df, Path(table_path),
                     overwrite=overwrite, index=False)
    if overall_table_path is not None:
        _maybe_write(overall_analytics, Path(overall_table_path),
                     overwrite=overwrite, index=False)

    return analytics_df, overall_analytics



# Analysis for each cluster and event time
def analyze_templates_and_waves(
        results_dir,
        save_dir,
        binary_path,
        ops,
        spike_times,
        spike_clusters,
        clusters_of_interest,
        selected_event_times,
        selected_event_indices,
        window_seconds,
        custom_nt,
        custom_nt0min,
        wave_time_range_wave,
        wave_time_range,
        temp_pad_length,
        wt_kwargs
):
    import pandas as pd
    import scipy.stats
    from scipy.signal import resample
    from scipy.signal import find_peaks
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import torch
    from kilosort.io import load_ops, BinaryFiltered
    from kilosort.data_tools import get_spike_waveforms, get_best_channels, get_best_channel  # get_templates

    final_df = pd.DataFrame()
    filtered_event_times = np.array(selected_event_times)[selected_event_indices]
    events_b_wave_counts = []
    events_a_wave_counts = []

    def get_templates(spike_idx, results_dir):
        results_dir = Path(results_dir)
        global counter
        if isinstance(spike_idx, int):
            spike_idx = [spike_idx]
        templates = np.load(results_dir / 'templates.npy')
        # Note that spike_clusters.npy is identical to spike_templates.npy for KS4
        spike_templates = np.load(results_dir / 'spike_clusters.npy')
        amplitudes = np.load(results_dir / 'amplitudes.npy')
        template_idx = spike_templates[spike_idx]
        temps = templates[template_idx, :, :]
        scaled = amplitudes[spike_idx, np.newaxis, np.newaxis] * temps
        return scaled

    def cluster_templates(cluster_id, results_dir, mean=False, best=False, spike_subset=None):
        results_dir = Path(results_dir)
        spike_clusters = np.load(results_dir / 'spike_clusters.npy')
        spike_idx = (spike_clusters == cluster_id).nonzero()[0]
        if spike_subset is not None:
            spike_idx = spike_idx[spike_subset]
        temps = get_templates(spike_idx, results_dir)
        if best:
            chan = get_best_channel(results_dir, cluster_id)
            temps = temps[:,:,chan]
        if mean:
            return temps.mean(axis=0)
        return temps

    def open_binary(binary_path=binary_path, custom_nt=None, custom_nt0min=None):
        bfile = bfile_from_ops_cust(ops=ops, filename=binary_path, nt_cust=custom_nt, nt0min_cust=custom_nt0min)
        return bfile

    def bfile_from_ops_cust(ops=None, ops_path=None, filename=None, device=None, nt_cust=None, nt0min_cust=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        if ops is None:
            if ops_path is not None:
                ops = load_ops(ops_path, device=device)
            else:
                raise ValueError('Must specify either `ops` or `ops_path`.')

        if filename is None:
            # Separate so that the same settings can be applied to the same binary
            # file at a new path, like when running pytests.
            filename = ops['filename']

        nt = nt_cust if nt_cust is not None else ops['nt']
        nt0min = nt0min_cust if nt0min_cust is not None else ops['nt0min']
        bfile = BinaryFiltered(
            filename, ops['n_chan_bin'], fs=ops['fs'], NT=ops['batch_size'],
            nt=nt, nt0min=nt0min, chan_map=ops['probe']['chanMap'],
            hp_filter=ops['fwav'], whiten_mat=ops['Wrot'], dshift=ops['dshift'],
            device=device, do_CAR=ops['do_CAR'], artifact_threshold=ops['artifact_threshold'],
            invert_sign=ops['invert_sign'], dtype=ops['data_dtype'], tmin=ops['tmin'],
            tmax=ops['tmax'], shift=ops['shift'], scale=ops['scale']
            )

        return bfile

    def get_spike_waveforms_cust(spikes, results_dir, bfile=None, chan=None):
        if isinstance(spikes, int):
            spikes = [spikes]

        #if bfile is None:
        #    ops = io.load_ops(results_dir / 'ops.npy')
        #    bfile = io.bfile_from_ops(ops)
        whitening_mat_inv = np.load(results_dir / 'whitening_mat_inv.npy')
        waves = []
        for t in spikes:
            #nt0min = 20 # nt0min  = time before spike, nt = time after spike + time before spike + 1
            #nt = 41
            #tmin = max(t - nt0min, 0)
            #tmax = t + (nt - nt0min)
            tmin = max(t - bfile.nt0min, 0)
            tmax = t + (bfile.nt - bfile.nt0min)

            w = bfile[tmin:tmax].cpu().numpy()
            if whitening_mat_inv is not None:
                w = whitening_mat_inv @ w
            if w.shape[1] == bfile.nt:
            #if w.shape[1] == nt:
                # Don't include spikes at the start or end of the recording that
                # get truncated to fewer time points.
                waves.append(w)
        waves = np.stack(waves, axis=-1)

        if chan is not None:
            waves = waves[chan,:]

        return waves

    def preprocess_template(template_pair):
          # Initial state of the input
        template = None
        # Check if template_pair is a list and has exactly two elements
        if isinstance(template_pair, list) and len(template_pair) == 2:
            # Check if the second element of the list is an array or a list
            if isinstance(template_pair[1], (list, np.ndarray)):
                # Convert the second element to a numpy array if it isn't already
                template = np.array(template_pair[1], dtype=float)
                return template
            else:
                print("Error: Second element of template_pair is neither a list nor an array.")
                return None
        elif isinstance(template_pair, np.ndarray) and template_pair.ndim == 1:
            # If template_pair is already a one-dimensional numpy array
            #print(f"Already numpy array: {template_pair}")
            template = template_pair
            return template
        else:
            print("Error: template_pair format is not recognized.")
            return None

        # Check if the template is empty
        if template.size == 0:
            print("Error: The template is empty.")
            return None

    def upsample_waveform(waveform, factor):
        """
        Upsample the waveform by a given factor using spline interpolation.
        """
        factor = 10
        if factor == 1:
            return waveform
        waveform = np.array(waveform)  # Ensure waveform is an array
        number_of_points = len(waveform) * factor
        return resample(waveform, number_of_points)

    def get_trough_and_peak_idx(template_pair):
        # Calculate trough and peak indices
        template = preprocess_template(template_pair)
        if template is None or len(template) == 0:
            print("Error: Invalid or empty template.")
            return None, None
        trough_idx = np.argmin(template)
        peak_idx = trough_idx + np.argmax(template[trough_idx:])
        return trough_idx, peak_idx

    #counter = 0
    def get_peak_to_valley(template_pair, sampling_frequency):
        # Check if the input is a list with exactly two elements
        global counter
        template = preprocess_template(template_pair)
        trough_idx, peak_idx = get_trough_and_peak_idx(template)
        #print(f'counter: {counter} trough_idx{trough_idx}, peak_idx{peak_idx}')
        if trough_idx is None or peak_idx is None:
            print("Failed to find valid trough and peak indices.")
            return np.nan

        if peak_idx <= trough_idx:
            print("Error: Peak index is not greater than trough index.")
            return np.nan

        peak_to_valley_time = (peak_idx - trough_idx) / sampling_frequency
        #print(f"Peak to Valley duration: {peak_to_valley_time} seconds.")
        #counter += 1
        return peak_to_valley_time

    def get_peak_trough_ratio(template_pair, sampling_frequency, trough_idx=None, peak_idx=None):
        template = preprocess_template(template_pair)
        trough_idx, peak_idx = get_trough_and_peak_idx(template)

        if template[trough_idx] == 0:
                print('template[trough_idx] == 0')
                return np.nan
        #print(f'template[peak_idx] {template[peak_idx]}, template[trough_idx]: {template[trough_idx]}')
        return template[peak_idx] / template[trough_idx]


    def get_half_width(template_pair, sampling_frequency, trough_idx=None, peak_idx=None, **wt_kwargs):

        template = preprocess_template(template_pair)
        trough_idx, peak_idx = get_trough_and_peak_idx(template)

        if not (0 <= trough_idx < len(template)) or not (0 <= peak_idx < len(template)):
            print("Error: Trough or peak index out of bounds.")
            return np.nan
        #print(f'trough_idx: {trough_idx}, peak_idx: {peak_idx}')
        trough_val = template[trough_idx]
        threshold = 0.5 * trough_val
        #print(f'threshold: {threshold}')
        cpre_idx = np.where(template[:trough_idx] <= threshold)[0]
        #print(f'cpre_idx: {cpre_idx}')
        cpost_idx = np.where(template[trough_idx:] <= threshold)[0]
        #print(f'cpre_idx: {cpost_idx}')
        if cpre_idx.size == 0 or cpost_idx.size == 0:
            print('Error: Threshold crossing indices not found.')
            return np.nan

        cross_pre_pk = cpre_idx[0] - 1 # last occurrence before trough
        cross_post_pk = cpost_idx[-1] + 1 + trough_idx  # first occurrence after trough
        hw = (cross_post_pk - cross_pre_pk) / sampling_frequency
        #print(f'hw {hw}')
        return hw

    def get_repolarization_slope(template_pair, sampling_frequency, **wt_kwargs):
        try:
            template = preprocess_template(template_pair)
            trough_idx, _ = get_trough_and_peak_idx(template)
            #print(f'trough_idx: {trough_idx}')
            rtrn_idx = np.nonzero(template[trough_idx:] >= 0)[0]
            #print(f'rtrn_idx: {rtrn_idx}')
            if len(rtrn_idx) == 0:
                print('len(rtrn_idx) == 0:')
                return np.nan
            return_to_base_idx = rtrn_idx[0] + trough_idx
            #print(f'return_to_base_idx: {return_to_base_idx}')
            if return_to_base_idx - trough_idx < 3:
                print('return_to_base_idx - trough_idx < 3')
                return np.nan
            times = np.arange(template.shape[0]) / sampling_frequency
            if np.std(times[trough_idx:return_to_base_idx]) == 0:
                print("Standard deviation of times is zero. Cannot perform regression.")
                return np.nan  # Or handle it by skipping this iteration or setting default values
            res = scipy.stats.linregress(times[trough_idx:return_to_base_idx], template[trough_idx:return_to_base_idx])
            return res.slope
        except Exception as e:
            print(f'error, scipy.stats repolarization: {e}')
        #print(f'res {res}')


    def get_recovery_slope(template_pair, sampling_frequency, peak_idx=None, **wt_kwargs):
        try:
            #print('get_recovery_slope active')
            template = preprocess_template(template_pair)
            _, peak_idx = get_trough_and_peak_idx(template)
            #print(f"grs_peak_idx: {peak_idx}")
            if peak_idx is None:
                print("Peak index not found.")
                return np.nan

            if not (0 <= peak_idx < len(template)):
                print("Invalid peak index.")
                return np.nan

            times = np.arange(template.shape[0]) / sampling_frequency
            if peak_idx == 0 or peak_idx >= len(template) - 1:
                print('Error: Invalid peak index or insufficient data for analysis.')
                return np.nan
            recovery_window_ms = wt_kwargs.get('recovery_window_ms')
            #print(f'recovery_window_ms {recovery_window_ms}')
            max_idx = int(peak_idx + ((recovery_window_ms / 1000) * sampling_frequency))
            max_idx = min(max_idx, len(template) - 1)  # Ensure not indexing out of bounds

            # Calculating the slope using linear regression
            if peak_idx < max_idx:
                res = scipy.stats.linregress(times[peak_idx:max_idx], template[peak_idx:max_idx])
                #print(f'res {res}')
                return res.slope
            else:
                print('Error: Peak index calculation went wrong or inadequate data range for slope calculation.')
                return np.nan
        except Exception as e:
            print(f'error, scipy.stats recovery: {e}')
            return np.nan


    def get_num_positive_peaks(template_pair, sampling_frequency, **wt_kwargs):
        template = preprocess_template(template_pair)
        value = wt_kwargs.get('peak_relative_threshold')
        #print(f'prt, {value}')

        peak_relative_threshold = wt_kwargs.get("peak_relative_threshold")
        peak_width_ms = wt_kwargs.get("peak_width_ms")
        max_value = np.max(np.abs(template))
        peak_width_samples = int(peak_width_ms / 1000 * sampling_frequency)

        pos_peaks = find_peaks(template, height=peak_relative_threshold * max_value, width=peak_width_samples)
        #print(f'pos_peaks {len(pos_peaks)}')
        return len(pos_peaks[0])

    def get_num_negative_peaks(template_pair, sampling_frequency, **kwargs):
        template = preprocess_template(template_pair)

        peak_relative_threshold = wt_kwargs.get("peak_relative_threshold")
        peak_width_ms = wt_kwargs.get("peak_width_ms")
        max_value = np.max(np.abs(template))
        peak_width_samples = int(peak_width_ms / 1000 * sampling_frequency)

        neg_peaks = find_peaks(-template, height=peak_relative_threshold * max_value, width=peak_width_samples)
        #print(f'neg_peaks {len(neg_peaks)}')
        return len(neg_peaks[0])

    def process_waveforms(waveforms, sampling_frequency, upsample_factor=10, **wt_kwargs):
        metrics = []
        for index, wave in enumerate(waveforms):
            wave = upsample_waveform(wave, upsample_factor)
            row_metrics = {}
            waveform_list = [index, wave]
            # Add function-specific kwargs adjustments as needed
            #print(f'idx{index}: mean {wave[1].mean()}')
            metric_functions = [
                (get_peak_to_valley, waveform_list, {}),
                (get_peak_trough_ratio, waveform_list, {}),
                (get_half_width, waveform_list, {}),
                (get_repolarization_slope, waveform_list, {'recovery_window_ms': wt_kwargs['recovery_window_ms']}),
                (get_recovery_slope, waveform_list, {'recovery_window_ms': wt_kwargs['recovery_window_ms']}),
                (get_num_positive_peaks, waveform_list, {'peak_relative_threshold': wt_kwargs['peak_relative_threshold'], 'peak_width_ms': wt_kwargs['peak_width_ms']}),
                (get_num_negative_peaks, waveform_list, {'peak_relative_threshold': wt_kwargs['peak_relative_threshold'], 'peak_width_ms': wt_kwargs['peak_width_ms']})
            ]

            for function, args, func_kwargs in metric_functions:
                row_metrics[function.__name__] = function(args, sampling_frequency, **func_kwargs)
            metrics.append(row_metrics)
        return pd.DataFrame(metrics)

    def process_cluster(cluster_id, before_temps, after_temps, before_waves, after_waves, sampling_frequency, **wt_kwargs):
        # Process each type of data
        df_before_temps = process_waveforms(before_temps, sampling_frequency, **wt_kwargs)
        #print(f'df_before_temps: {df_before_temps}')
        df_after_temps = process_waveforms(after_temps, sampling_frequency, **wt_kwargs)
        #print(f'df_after_temps: {df_after_temps}')
        df_before_waves = process_waveforms(before_waves, sampling_frequency, **wt_kwargs)
        #print(f'df_before_waves: {df_before_waves}')
        df_after_waves = process_waveforms(after_waves, sampling_frequency, **wt_kwargs)
        #print(f'df_after_waves: {df_after_waves}')

        # Combine all dataframes column-wise
        combined_df = pd.concat([df_before_temps, df_after_temps, df_before_waves, df_after_waves], axis=1)
        combined_df['cluster_id'] = cluster_id  # Add cluster ID to the combined dataframe
        return combined_df

    for cluster_id in clusters_of_interest:
        cluster_spike_indices = (spike_clusters == cluster_id).nonzero()[0]
        cluster_spike_samples = spike_times[cluster_spike_indices]
        cluster_spike_times = cluster_spike_samples / ops['fs']# Convert to seconds
        events_b_waves = []
        events_a_waves = []
        before_events_spike_sec = []
        after_events_spike_sec = []
        before_events_cluster_spike_indices = []
        after_events_cluster_spike_indices = []
        total_b_waves = 0
        total_a_waves = 0

        for event_time in filtered_event_times:
            # Define the window around each event time
            start_window = event_time - window_seconds
            start_window_sample = int(start_window * ops['fs'])
            mid_window = event_time
            mid_window_sample = int(mid_window * ops['fs'])
            end_window = event_time + window_seconds
            end_window_sample = int(end_window * ops['fs'])

            # Get indices of spikes within the window around the event time
            before_event_cluster_spike_sample_indices = np.where((cluster_spike_samples >= start_window_sample) & (cluster_spike_samples < mid_window_sample))[0]
            after_event_cluster_spike_sample_indices = np.where((cluster_spike_samples >= mid_window_sample) & (cluster_spike_samples <= end_window_sample))[0]

            # event_spike_times_indices: indices relative to spike_times.npy
            before_event_cluster_spike_samples = cluster_spike_samples[before_event_cluster_spike_sample_indices]
            after_event_cluster_spike_samples = cluster_spike_samples[after_event_cluster_spike_sample_indices]

            before_event_spike_sec = cluster_spike_samples[before_event_cluster_spike_sample_indices] / ops['fs']  # Convert to seconds
            after_event_spike_sec = cluster_spike_samples[after_event_cluster_spike_sample_indices] / ops['fs']  # Convert to seconds

            # Extend the lists with the indices and times of spikes around each event
            before_events_cluster_spike_indices.extend(before_event_cluster_spike_sample_indices)
            after_events_cluster_spike_indices.extend(after_event_cluster_spike_sample_indices)
            before_events_spike_sec.extend(before_event_spike_sec)
            after_events_spike_sec.extend(after_event_spike_sec)
            best_clusID_chan = get_best_channels(kilosort_sorter_output_folder)[cluster_id]

            b_waves = np.empty((0, 0))
            if len(before_event_cluster_spike_samples) > 0:
                bfile = open_binary(custom_nt = custom_nt if custom_nt is not None else None, custom_nt0min = custom_nt0min if custom_nt0min is not None else None)
                b_waves = get_spike_waveforms_cust(
                    spikes=before_event_cluster_spike_samples,           # In samples
                    results_dir=results_dir,           # Path to Kilosort output folder
                    bfile=bfile,                       # Optional, but recommended
                    chan=best_clusID_chan                          # Optional channel index (int) or None for all
                )
                events_b_waves.append(b_waves)
            else:
                events_b_waves.append(np.empty((0, 0)))

            a_waves = np.empty((0, 0))
            if len(after_event_cluster_spike_samples) > 0:
                bfile = open_binary(custom_nt = custom_nt if custom_nt is not None else None, custom_nt0min = custom_nt0min if custom_nt0min is not None else None)
                a_waves = get_spike_waveforms_cust(
                    spikes=after_event_cluster_spike_samples,           # In samples
                    results_dir=results_dir,           # Path to Kilosort output folder
                    bfile=bfile,                       # Optional, but recommended
                    chan=best_clusID_chan                          # Optional channel index (int) or None for all
                )
                events_a_waves.append(a_waves)
            else:
                events_a_waves.append(np.empty((0, 0)))
            # Count waves if they exist and append to respective lists
            num_b_waves = b_waves.shape[1] if b_waves.ndim > 1 and b_waves.size > 0 else 0
            num_a_waves = a_waves.shape[1] if a_waves.ndim > 1 and a_waves.size > 0 else 0
            events_b_wave_counts.append(num_b_waves)
            events_a_wave_counts.append(num_a_waves)

        before_all_temp = np.empty((0, 0))
        before_all_temp = cluster_templates(cluster_id, kilosort_sorter_output_folder, mean=False, best=True, spike_subset=before_events_cluster_spike_indices)

        before_mean_temp = np.empty((0, 0))
        before_mean_temp = cluster_templates(cluster_id, kilosort_sorter_output_folder, mean=True, best=True, spike_subset=before_events_cluster_spike_indices)

        after_all_temp = np.empty((0, 0))
        after_all_temp = cluster_templates(cluster_id, kilosort_sorter_output_folder, mean=False, best=True, spike_subset=after_events_cluster_spike_indices)

        after_mean_temp = np.empty((0, 0))
        after_mean_temp = cluster_templates(cluster_id, kilosort_sorter_output_folder, mean=True, best=True, spike_subset=after_events_cluster_spike_indices)

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 6))
        fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 6))
        num_before_waves = sum(1 for w in events_b_waves if w.size > 0)
        num_after_waves = sum(1 for w in events_a_waves if w.size > 0)

        if sum(events_b_wave_counts) > 0:
            before_all_waves = np.concatenate([wave_set for wave_set in events_b_waves if wave_set.size > 0], axis=1)
            mean_waveform_before = np.empty((0, 0))
            mean_waveform_before = np.mean(before_all_waves, axis=-1)
            b_sec_index = 0
            for wave_set in events_b_waves:
                for i in range(wave_set.shape[1]):
                    ax1.plot(wave_time_range_wave, [before_events_spike_sec[i+b_sec_index]], zs=wave_set[:, i], zdir='z')
                    ax5.plot(wave_time_range_wave, wave_set[:, i])
                b_sec_index += wave_set.shape[1]

            # Plot the mean waveform computed from individual waveforms
            for j in range(before_all_temp.shape[0]):
                padded_before_template = [None] * temp_pad_length + list(before_all_temp[j, :])
                ax3.plot(wave_time_range, [before_events_spike_sec[j]],zs=before_mean_temp.flatten(), zdir='z')
                ax7.plot(wave_time_range, before_all_temp[j,:])

            if sum(events_b_wave_counts) > 1:
                avg_bess = np.mean(before_events_spike_sec)
                padded_before_mean_template = [None] * temp_pad_length + list(before_mean_temp.flatten())
                ax1.plot(wave_time_range_wave, [avg_bess], zs=mean_waveform_before, zdir='z', color='black', linewidth=4, label='Mean Waveform')
                # Plot the mean template waveform
                ax1.plot(wave_time_range_wave, [avg_bess], zs=padded_before_mean_template, zdir='z', color='red', linewidth=4, label='Mean Template')
                ax1.legend()  # Add legend to identify the mean template
                ax3.plot(wave_time_range, [avg_bess], zs=before_mean_temp.flatten(), zdir='z', color='red', linewidth=4, label='Mean Template')
                ax3.legend()
                ax5.plot(wave_time_range_wave, mean_waveform_before, color='black', linewidth=4, label='Mean Waveform')
                ax5.plot(wave_time_range_wave, padded_before_mean_template, color='red', linewidth=4, label='Mean Template')
                ax5.legend()
                ax7.plot(wave_time_range, before_mean_temp.flatten(), color='red', linewidth=4, label='Mean Template')
                ax7.legend()
            elif sum(events_b_wave_counts) > 0: # Plot the individual template waveform
                padded_before_mean_template = [None] * temp_pad_length + list(before_mean_temp.flatten())
                ax1.plot(wave_time_range_wave, [before_events_spike_sec], zs=mean_waveform_before, zdir='z', color='black', linewidth=4, label='Individual Template')
                ax1.plot(wave_time_range_wave, [before_events_spike_sec], zs=before_mean_temp.flatten(), zdir='z', color='red', linewidth=4, label='Individual Template')
                ax1.legend()
                ax3.plot(wave_time_range, [before_events_spike_sec], zs=before_mean_temp.flatten(), zdir='z', color='red', linewidth=4, label='Individual Template')
                ax3.legend()
                ax5.plot(wave_time_range_wave, padded_before_mean_template, color='red', linewidth=4, label='Individual Template')
                ax5.legend()
                ax7.plot(wave_time_range, before_mean_temp.flatten(), color='red', linewidth=4, label='Individual Template')
                ax7.legend()
                # Add legend to identify the mean template
        else: # Plot an empty graph or display a message
                ax1.text(0.5, 0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center')
                ax1.set_title(f'After Event Spikes (ClusID {cluster_id})')

        ax1.set_xlabel('Time (ms)', fontsize=14, labelpad=15)
        ax1.set_ylabel('Spike time (s)', fontsize=14, labelpad=15)
        ax1.set_zlabel('Amplitude (uV)', fontsize=14, labelpad=15)
        ax1.set_title(f'Before Event Unwhitened Waves (ClusID {cluster_id})', fontsize=14)
        ax1.view_init(azim=-100, elev=20)
        ax3.set_xlabel('Time (ms)', fontsize=14, labelpad=15)
        ax3.set_ylabel('Spike time (s)', fontsize=14, labelpad=15)
        ax3.set_zlabel('Amplitude (uV)', fontsize=14, labelpad=15)
        ax3.set_title(f'Before Event Scaled Templates (ClusID {cluster_id})', fontsize=17)
        ax3.view_init(azim=-100, elev=20)
        ax5.set_xlabel('Spike Time (ms)', fontsize=14, labelpad=15)
        ax5.set_ylabel('Amplitude (uV)', fontsize=14, labelpad=15)
        ax5.set_title(f'Before Event Unwhitened Waves (ClusID {cluster_id})', fontsize=17)
        ax7.set_xlabel('Spike Time (ms)', fontsize=14, labelpad=15)
        ax7.set_ylabel('Amplitude (uV)', fontsize=14, labelpad=15)
        ax7.set_title(f'Before Event Scaled Templates (ClusID {cluster_id})', fontsize=17)
        ax7.set_ylim([-150, 40])

        if sum(events_a_wave_counts) > 0 :
            after_all_waves = np.concatenate([wave_set for wave_set in events_a_waves if wave_set.size > 0], axis=1)
            mean_waveform_after = np.empty((0, 0))
            mean_waveform_after = np.mean(after_all_waves, axis=-1)
            a_sec_index = 0
            for wave_set in events_a_waves:
                for k in range(wave_set.shape[1]):
                    ax2.plot(wave_time_range_wave, [after_events_spike_sec[k+a_sec_index]], zs=wave_set[:, k], zdir='z')
                    ax6.plot(wave_time_range_wave, wave_set[:, k])
                a_sec_index += wave_set.shape[1]

            # Plot the mean waveform computed from individual waveforms
            for l in range(after_all_temp.shape[0]):
                padded_after_template = [None] * temp_pad_length + list(after_all_temp[l, :])
                ax4.plot(wave_time_range, [after_events_spike_sec[l]],zs=after_all_temp[l, :], zdir='z')
                ax8.plot(wave_time_range, after_all_temp[l, :])

            if sum(events_a_wave_counts)>1:
                avg_aess = np.mean(after_events_spike_sec)
                padded_after_mean_template = [None] * temp_pad_length + list(after_mean_temp.flatten())
                ax2.plot(wave_time_range_wave, [avg_aess], zs=mean_waveform_after, zdir='z', color='black', linewidth=4, label='Mean Waveform')
                ax2.plot(wave_time_range_wave, [avg_aess], zs=padded_after_mean_template, zdir='z', color='red', linewidth=4, label='Mean Template')
                ax2.set_zlim([-500, 300])
                ax2.legend()
                ax4.plot(wave_time_range, [avg_aess], after_mean_temp.flatten(), color='red', linewidth=4, label='Mean Template')
                ax4.legend()# Add legend to identify the mean template
                ax6.plot(wave_time_range_wave, mean_waveform_after, color='black', linewidth=4, label='Mean Waveform')
                ax6.plot(wave_time_range_wave, padded_after_mean_template, color='red', linewidth=4, label='Mean Template')
                ax6.legend()
                ax8.plot(wave_time_range, after_mean_temp.flatten(), color='red', linewidth=4, label='Mean Template')
                ax8.legend()

            elif sum(events_a_wave_counts)>0: # Plot the mean template waveform
                padded_after_mean_template = [None] * temp_pad_length + list(after_mean_temp.flatten())
                ax2.plot(wave_time_range_wave, [after_events_spike_sec], zs=mean_waveform_after, zdir='z', color='black', linewidth=4, label='Individual Waveform')
                ax2.plot(wave_time_range_wave, [after_events_spike_sec], zs=padded_after_mean_template, zdir='z', color='red', linewidth=4, label='Individual Template')
                ax2.legend()
                ax4.plot(wave_time_range_wave, [after_events_spike_sec], zs=padded_after_mean_template, zdir='z', color='red', linewidth=4, label='Individual Template')
                ax4.legend()
                ax6.plot(wave_time_range_wave, mean_waveform_after, color='black', linewidth=4, label='Individual Waveform')
                ax6.plot(wave_time_range_wave, padded_after_mean_template, color='red', linewidth=4, label='Individual Template')
                ax6.legend()
                ax8.plot(wave_time_range, after_mean_temp.flatten(), color='red', linewidth=4, label='Individual Template')
                ax8.legend()
                # Add legend to identify the mean template
        else: # Plot an empty graph or display a message
                ax2.text(0.5, 0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center')
                ax2.set_title(f'After Event Spikes (ClusID {cluster_id})')
        ax2.set_xlabel('Time (ms)', fontsize=14, labelpad=15)
        ax2.set_ylabel('Spike time (s)', fontsize=14, labelpad=15)
        ax2.set_zlabel('Amplitude (uV)', fontsize=14, labelpad=15)
        ax2.set_title(f'After Event Unwhitened Waves (ClusID {cluster_id})', fontsize=14)
        ax2.view_init(azim=-100, elev=20)
        ax4.set_xlabel('Time (ms)', fontsize=14, labelpad=15)
        ax4.set_ylabel('Spike time (s)', fontsize=14, labelpad=15)
        ax4.set_zlabel('Amplitude (uV)', fontsize=14, labelpad=15)
        ax4.set_title(f'After Event Scaled Templates (ClusID {cluster_id})', fontsize=17)
        ax4.view_init(azim=-100, elev=20)
        ax6.set_xlabel('Spike time (s)', fontsize=14, labelpad=15)
        ax6.set_ylabel('Amplitude (uV)', fontsize=14, labelpad=5)
        ax6.set_title(f'After Event Unwhitened Waves (ClusID {cluster_id})', fontsize=14)
        ax8.set_xlabel('Spike time (s)', fontsize=14, labelpad=15)
        ax8.set_ylabel('Amplitude (uV)', fontsize=14, labelpad=10)
        ax8.set_title(f'After Event Scaled Temps (ClusID {cluster_id})', fontsize=14)
        ax8.set_ylim([-150, 40])

        # Adjust the layout and show the plot
        plt.tight_layout()  # Adjust the layout to make sure there's no overlap
        fig_name = save_dir / f'clusID-{cluster_id}_psw_3d_all'
        plt.savefig(save_dir,)
        plt.show()

        # Calculate wave and template metrics
        #print(f"check dimensions: {before_all_temp.shape}, {after_all_temp.shape}, {before_all_waves.T.shape}, {after_all_waves.T.shape}")
        cluster_results_df = process_cluster(cluster_id, before_all_temp, after_all_temp, before_all_waves.T, after_all_waves.T, 30000, **wt_kwargs)
        if cluster_results_df.isnull().values.any():
            print(f"Debug: NaN values found for cluster_id {cluster_id}")
        final_df = pd.concat([final_df, cluster_results_df], ignore_index=True)

    # New column names list
    new_column_names = [
        'b-temp_peak_to_valley',	'b-temp_get_peak_trough_ratio',	'b-temp_get_half_width',	'b-temp_get_repolarization_slope',	'b-temp_get_recovery_slope',	'b-temp_get_num_positive_peaks',	'b-temp_get_num_negative_peaks',
    'a-temp_peak_to_valley',	'a-temp_get_peak_trough_ratio',	'a-temp_get_half_width',	'a-temp_get_repolarization_slope',	'a-temp_get_recovery_slope',	'a-temp_get_num_positive_peaks',	'a-temp_get_num_negative_peaks',
        'b-wave_peak_to_valley',	'b-wave_get_peak_trough_ratio',	'b-wave_get_half_width',	'b-wave_get_repolarization_slope',	'b-wave_get_recovery_slope',	'b-wave_get_num_positive_peaks',	'b-wave_get_num_negative_peaks',
        'a-wave_peak_to_valley',	'a-wave_get_peak_trough_ratio',	'a-wave_get_half_width',	'a-wave_get_repolarization_slope',	'a-wave_get_recovery_slope',	'a-wave_get_num_positive_peaks',	'a-wave_get_num_negative_peaks', '']

    # Replacing the old column names with the new ones from the list
    final_df.columns = new_column_names

    final_df.to_csv(custom_analytics_folder / 'custom_temp_wave_metrics_1.csv')
    print(final_df.head())

def save_spike_indices_in_windows(event_times, pre_window, post_window, sampling_rate, sorting, output_filename):
    spikes = sorting.to_spike_vector(concatenated=True)
    spike_indices = []

    for event_time in event_times:
        start_sample_unit = int((event_time + pre_window) * sampling_rate)
        end_sample_unit = int((event_time + post_window) * sampling_rate)

        # Find indices of spikes within the range [start_sample_unit, end_sample_unit]
        valid_indices = np.where((spikes['sample_index'] >= start_sample_unit) &
                                 (spikes['sample_index'] <= end_sample_unit))[0]

        spike_indices.append(valid_indices)

    # Save spike indices to a file for later use as a custom mask
    spike_indices = np.concatenate(spike_indices)  # Flatten the list of arrays to a single array if necessary
    np.save(output_filename, spike_indices)

    return spike_indices

# ---------------------------------------------------------------------
#  Neuropixel_analysis_module.py
#  (only the function below is new/updated)
# ---------------------------------------------------------------------
def analyze_burst_spikes(
        clusters_of_interest,
        spike_clusters,
        spike_times,
        spike_times_sec,
        amplitudes,
        non_event_mask,
        thr_burst,
        thr_single,
        burst_type_path,
        burst_analysis_summary,
        overwrite: bool = False        # ← NEW
):
    """
    Tag spikes as single vs burst, build diagnostics + summary table,
    and save JSONs/CSV.  File operations honour *overwrite* and emit
    concise log messages that Voilà can capture.

    Returns
    -------
    burst_spike_groups : dict[int, list[list[int]]]
        Cluster-ID → list of bursts (each burst = list of spike indices)
    single_spike_indices : dict[int, ndarray[int]]
        Cluster-ID → 1-D array of single-spike indices
    """
    # ───────────────────────── imports & utilities ─────────────────────
    import numpy as np
    import pandas as pd
    import json, matplotlib.pyplot as plt
    from scipy.stats import sem, shapiro, wilcoxon, ttest_rel, friedmanchisquare
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from pathlib import Path
    from matplotlib.lines import Line2D

    # ---------- tiny helper: safe writer + log ------------------------
    def _maybe_write(obj, dst: Path, *, overwrite: bool, **kw):
        """
        Save *obj* to *dst* while respecting *overwrite* and printing
        what happened (for Voilà log capture).

        • pandas.DataFrame → .to_csv
        • matplotlib Figure → .savefig
        • dict/list/str/…   → JSON or raw text
        """
        dst = Path(dst)
        exists_before = dst.exists()
        if exists_before and not overwrite:
            print(f"Skipped (exists, overwrite=False): {dst}")
            return

        dst.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(obj, "to_csv"):
            obj.to_csv(dst, **kw)
        elif hasattr(obj, "savefig"):
            obj.savefig(dst, **kw)
        else:                                  # plain text / JSON
            if isinstance(obj, (dict, list)):
                dst.write_text(json.dumps(obj, indent=2))
            else:
                dst.write_text(str(obj))

        verb = "Over-written" if exists_before and overwrite else "Saved"
        print(f"{verb} → {dst}")

    # ---------- misc helpers (unchanged logic) ------------------------
    def safe_sem(x):  return sem(x) if len(x) > 1 else np.nan

    def safe_shapiro(x):
        if len(x) < 3:
            return False, np.nan
        stat, p = shapiro(x)
        return (p > 0.05), p

    # ------------------------------------------------------------------
    #  run_stat_tests_for_bursts — exactly as in your original code
    # ------------------------------------------------------------------
    def run_stat_tests_for_bursts(bursts, amps, alpha=0.05):
        import pandas as pd
        # build long-form DF: burst × order
        rec = []
        for bid, b in enumerate(bursts):
            for order, idx in enumerate(b):
                rec.append({"burst": bid, "order": order, "amp": amps[idx]})
        df = pd.DataFrame(rec)
        levels = sorted(df.order.unique())
        if len(levels) < 2:
            return None, np.nan, None

        # normality per order
        normals = {o: (len(df[df.order == o]) >= 3 and
                       shapiro(df.loc[df.order == o, 'amp'])[1] > 0.05)
                   for o in levels}

        # two-level case
        if len(levels) == 2:
            wide = df.pivot(index="burst", columns="order", values="amp").dropna()
            x, y = wide.iloc[:, 0], wide.iloc[:, 1]
            if all(normals.values()):
                stat, p = ttest_rel(x, y)
                return "paired t-test", p, None
            stat, p = wilcoxon(x, y)
            return "Wilcoxon", p, None

        # >2 levels: all normal → RM-ANOVA + Tukey
        import statsmodels.api as sm
        if all(normals.values()):
            aov = AnovaRM(df, 'amp', 'burst', within=['order']).fit()
            p = aov.anova_table.loc['order', 'Pr > F']
            tuk = pairwise_tukeyhsd(df.amp, df.order.astype(str), alpha=alpha)
            return "RM-ANOVA", p, tuk

        # else: Friedman + pairwise Wilcoxon w/ Bonferroni
        pivot = df.pivot(index="burst", columns="order", values="amp").dropna()
        stat, p = friedmanchisquare(*[pivot[o] for o in levels])
        post = []
        for i, o1 in enumerate(levels):
            for o2 in levels[i+1:]:
                wstat, p_raw = wilcoxon(pivot[o1], pivot[o2])
                m = len(levels) * (len(levels)-1) / 2
                post.append({"order1": o1, "order2": o2,
                             "p_uncorrected": p_raw,
                             "p_bonf": min(p_raw*m, 1.0)})
        return "Friedman", p, pd.DataFrame(post)

    # ───────────────────────── main loop per cluster ───────────────────
    burst_spike_groups, single_spike_indices = {}, {}
    for cid in clusters_of_interest:
        mask = (spike_clusters == cid) & non_event_mask
        idx  = np.where(mask)[0]
        if idx.size == 0:
            burst_spike_groups[cid]   = []
            single_spike_indices[cid] = np.array([], int)
            continue

        order = np.argsort(spike_times[idx])
        idx_sorted = idx[order]
        t_sorted   = spike_times_sec[idx_sorted]

        isi_prev = np.diff(t_sorted, prepend=-np.inf)
        isi_next = np.diff(t_sorted, append = np.inf)
        in_burst = (isi_prev < thr_burst) | (isi_next < thr_burst)

        # contiguous sequences → bursts
        starts = np.flatnonzero(np.diff(np.r_[0, in_burst.astype(int)]) == 1)
        ends   = np.flatnonzero(np.diff(np.r_[in_burst.astype(int), 0]) == -1)
        bursts = [idx_sorted[s:e+1] for s, e in zip(starts, ends) if e+1-s >= 2]

        mark = np.zeros(len(idx_sorted), bool)
        for b in bursts:
            pos0 = np.searchsorted(idx_sorted, b[0])
            mark[pos0:pos0+len(b)] = True

        singles = idx_sorted[
            ~mark & (isi_prev > thr_single) & (isi_next > thr_single)
        ]

        burst_spike_groups[cid]   = bursts
        single_spike_indices[cid] = singles

    # ───────────────────────── JSON output per cluster ─────────────────
    burst_type_path = Path(burst_type_path)
    for cid, bursts in burst_spike_groups.items():
        _maybe_write([b.tolist() for b in bursts],
                     burst_type_path / f"burst_spike_groups_{cid}.json",
                     overwrite=overwrite, indent=2)

    for cid, singles in single_spike_indices.items():
        _maybe_write(singles.tolist(),
                     burst_type_path / f"single_spike_indices_{cid}.json",
                     overwrite=overwrite, indent=2)

    # ───────────────────────── summary DataFrame & CSV ─────────────────
    summary_rows = []

    for cid in clusters_of_interest:
        bursts  = burst_spike_groups[cid]
        singles = single_spike_indices[cid]

        lengths = np.array([len(b) for b in bursts])
        n_bursts, n_singles = len(bursts), len(singles)
        n_burst_spikes      = lengths.sum()

        # ---------- amplitude by order ---------------------------------
        amp_by_order = {}
        for b in bursts:
            for o, idx in enumerate(b):
                amp_by_order.setdefault(o, []).append(amplitudes[idx])

        mean_amp = {o: np.mean(v) for o, v in amp_by_order.items()}
        sem_amp  = {o: safe_sem(v) for o, v in amp_by_order.items()}
        norm_ok  = {o: safe_shapiro(v)[0] for o, v in amp_by_order.items()}

        test_name, p_val, _ = run_stat_tests_for_bursts(bursts, amplitudes)

        row = dict(
            cluster_id = cid,
            n_single_spikes = n_singles,
            n_bursts        = n_bursts,
            n_burst_spikes  = n_burst_spikes,
            spike_amp_test  = test_name,
            spike_amp_pvalue= p_val,
            mean_spikes_per_burst = lengths.mean() if n_bursts else np.nan,
        )
        for o in range(max(mean_amp.keys(), default=-1)+1):
            row[f"Amplitude_Order_{o+1}"] = mean_amp.get(o, np.nan)
            row[f"SEM_Order_{o+1}"]       = sem_amp.get(o,  np.nan)
            row[f"Normal_Order_{o+1}"]    = norm_ok.get(o,  False)

        summary_rows.append(row)

    final_df = pd.DataFrame(summary_rows)
    core_cols = [
        "cluster_id", "n_single_spikes", "n_bursts", "n_burst_spikes",
        "spike_amp_test", "spike_amp_pvalue", "mean_spikes_per_burst"
    ]
    amp_cols  = sorted(c for c in final_df.columns if c.startswith("Amplitude_Order_"))
    sem_cols  = sorted(c for c in final_df.columns if c.startswith("SEM_Order_"))
    norm_cols = sorted(c for c in final_df.columns if c.startswith("Normal_Order_"))
    final_df  = final_df[core_cols + amp_cols + sem_cols + norm_cols]

    _maybe_write(final_df,
                 Path(burst_analysis_summary),
                 overwrite=overwrite, index=False)

    return burst_spike_groups, single_spike_indices


def load_burst_and_single_spikes(burst_type_path, units_of_interest):
    import json
    import numpy as np

    burst_spike_groups = {}
    single_spike_indices = {}
    for cid in units_of_interest:
        # Load and convert burst_spike_groups
        with open(burst_type_path / f'burst_spike_groups_{cid}.json', 'r') as f:
            # Load data as list of lists and convert each list to a NumPy array
            burst_lists = json.load(f)
            burst_spike_groups[cid] = [np.array(burst) for burst in burst_lists]

        # Load and convert single_spike_indices

        with open(burst_type_path / f'single_spike_indices_{cid}.json', 'r') as f:
            singles_list = json.load(f)
            single_spike_indices[cid] = np.array(singles_list)
    return burst_spike_groups, single_spike_indices


def analyze_burst_spikes2(
        clusters_of_interest,
        spike_clusters,
        spike_times,
        spike_times_sec,
        amplitudes,
        non_event_mask,
        thr_burst,
        thr_single,
        burst_type_path,
        burst_analysis_summary,
        overwrite: bool = True
):
    """
    Tag spikes as SINGLE vs BURST (per cluster), save per-cluster JSON artifacts,
    write a summary CSV, and generate DISTRIBUTION PLOTS — including
    size-specific amplitude plots AND raw point overlays on boxplots.
    """
    # ───────────────────────── imports & setup ─────────────────────────
    import numpy as np
    import pandas as pd
    import json
    from pathlib import Path
    from scipy.stats import sem, shapiro, wilcoxon, ttest_rel, friedmanchisquare
    import matplotlib.pyplot as plt

    # Optional deps (statsmodels) for RM-ANOVA/Tukey; fall back if missing
    try:
        from statsmodels.stats.anova import AnovaRM
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        _HAS_STATSMODELS = True
    except Exception:
        AnovaRM = None
        pairwise_tukeyhsd = None
        _HAS_STATSMODELS = False

    burst_type_path = Path(burst_type_path)
    burst_type_path.mkdir(parents=True, exist_ok=True)
    fig_dir = burst_type_path / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    burst_analysis_summary = Path(burst_analysis_summary)

    # ---------- tiny helpers: writers & plotting -----------------------
    def _maybe_write(obj, dst: Path, *, overwrite: bool, **kw):
        dst = Path(dst)
        exists_before = dst.exists()
        if exists_before and not overwrite:
            print(f"Skipped (exists, overwrite=False): {dst}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(obj, "to_csv"):
            obj.to_csv(dst, **kw)
        else:
            if isinstance(obj, (dict, list)):
                dst.write_text(json.dumps(obj, indent=2))
            else:
                dst.write_text(str(obj))
        verb = "Over-written" if exists_before and overwrite else "Saved"
        print(f"{verb} → {dst}")

    def _save_fig(fig, dst: Path, *, overwrite: bool, dpi: int = 200, bbox_inches="tight"):
        dst = Path(dst)
        exists_before = dst.exists()
        if exists_before and not overwrite:
            print(f"Skipped (exists, overwrite=False): {dst}")
            plt.close(fig)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dst, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        verb = "Over-written" if exists_before and overwrite else "Saved"
        print(f"{verb} → {dst}")

    def _safe_sem(x):
        x = np.asarray(x)
        return sem(x) if x.size > 1 else np.nan

    def _safe_shapiro(x):
        x = np.asarray(x)
        if x.size < 3:
            return False, np.nan
        stat, p = shapiro(x)
        return (p > 0.05), p

    # Overlay helper: jittered points over boxplots (caps max points for clarity/speed)
    def _overlay_scatter_on_box(ax, data_by_group, *, jitter=0.03, alpha=0.7,
                                s=18, max_points=20000, seed=123):
        """
        data_by_group: list of 1D arrays (group 1..G). x-positions assumed 1..G.
        """
        rng = np.random.default_rng(seed)
        total = int(sum(len(d) for d in data_by_group))
        # Proportional subsample if too many points
        if total > max_points:
            keep = max_points
            # target per group ~ proportional to size
            sizes = np.array([len(d) for d in data_by_group], dtype=float)
            if sizes.sum() == 0:
                return
            targets = np.floor(keep * sizes / sizes.sum()).astype(int)
            # fix rounding to exact total
            while targets.sum() < keep:
                inc = np.argmax(sizes - targets)
                targets[inc] += 1
        else:
            targets = np.array([len(d) for d in data_by_group], int)

        x_positions = []
        y_values = []
        for i, arr in enumerate(data_by_group, start=1):
            if len(arr) == 0:
                continue
            n_take = int(targets[i-1])
            if n_take == 0:
                continue
            if n_take < len(arr):
                idx = rng.choice(len(arr), size=n_take, replace=False)
                vals = np.asarray(arr)[idx]
            else:
                vals = np.asarray(arr)
            xj = i + rng.uniform(-jitter, +jitter, size=vals.size)
            x_positions.append(xj)
            y_values.append(vals)

        if x_positions:
            ax.scatter(np.concatenate(x_positions),
                       np.concatenate(y_values),
                       s=s, alpha=alpha, zorder=3, linewidths=0)

    # ------------------------------------------------------------------
    #  run_stat_tests_for_bursts — robust and dependency-aware
    # ------------------------------------------------------------------
    def run_stat_tests_for_bursts(bursts, amps, alpha=0.05):
        if not bursts or sum(len(b) for b in bursts) == 0:
            return None, np.nan, None

        rec = []
        for bid, b in enumerate(bursts):
            for order_idx, spike_idx in enumerate(b):
                rec.append({"burst": bid, "order": order_idx, "amp": float(amps[spike_idx])})
        if not rec:
            return None, np.nan, None

        df = pd.DataFrame.from_records(rec)
        if df.empty or "order" not in df.columns:
            return None, np.nan, None

        levels = sorted(df["order"].unique())
        if len(levels) < 2:
            return None, np.nan, None

        normals = {}
        for o in levels:
            vals = df.loc[df["order"] == o, "amp"].to_numpy()
            normals[o] = (vals.size >= 3 and shapiro(vals)[1] > 0.05)

        if len(levels) == 2:
            wide = df.pivot(index="burst", columns="order", values="amp").dropna()
            if wide.shape[0] < 2:
                return None, np.nan, None
            x, y = wide.iloc[:, 0].to_numpy(), wide.iloc[:, 1].to_numpy()
            if all(normals.values()):
                _, p = ttest_rel(x, y)
                return "paired t-test", float(p), None
            _, p = wilcoxon(x, y)
            return "Wilcoxon", float(p), None

        if all(normals.values()) and _HAS_STATSMODELS:
            aov = AnovaRM(df, "amp", "burst", within=["order"]).fit()
            p = float(aov.anova_table.loc("order")["Pr > F"]) if "order" in aov.anova_table.index else np.nan
            try:
                tuk = pairwise_tukeyhsd(df["amp"], df["order"].astype(str), alpha=alpha)
            except Exception:
                tuk = None
            return "RM-ANOVA", p, tuk

        pivot = df.pivot(index="burst", columns="order", values="amp").dropna()
        if pivot.shape[0] < 2:
            return None, np.nan, None
        _, p = friedmanchisquare(*[pivot[o] for o in levels])

        post = []
        m = len(levels) * (len(levels) - 1) // 2
        for i, o1 in enumerate(levels):
            for o2 in levels[i+1:]:
                wstat, p_raw = wilcoxon(pivot[o1], pivot[o2])
                post.append({
                    "order1": int(o1), "order2": int(o2),
                    "p_uncorrected": float(p_raw),
                    "p_bonf": float(min(p_raw * m, 1.0))
                })
        post_df = pd.DataFrame(post) if post else None
        return "Friedman", float(p), post_df

    # ───────────────────────── main loop per cluster ───────────────────
    burst_spike_groups, single_spike_indices = {}, {}

    for cid in clusters_of_interest:
        # Select spikes for this cluster and mask them
        use = (spike_clusters == cid) & non_event_mask
        idx = np.flatnonzero(use)

        if idx.size == 0:
            burst_spike_groups[cid] = []
            single_spike_indices[cid] = np.array([], dtype=int)
            _maybe_write([], burst_type_path / f"burst_spike_groups_{cid}.json",
                         overwrite=overwrite)
            _maybe_write([], burst_type_path / f"single_spike_indices_{cid}.json",
                         overwrite=overwrite)
            continue

        # Sort by time
        idx_sorted = idx[np.argsort(spike_times[idx])]
        t_sorted = spike_times_sec[idx_sorted]
        a_sorted = amplitudes[idx_sorted] if amplitudes is not None else None

        # Neighbor ISIs (seconds)
        isi_prev = np.diff(t_sorted, prepend=-np.inf)
        isi_next = np.diff(t_sorted, append=np.inf)

        # Burst membership: any neighbor within thr_burst
        in_burst = (isi_prev < thr_burst) | (isi_next < thr_burst)

        # Find contiguous in_burst runs; keep runs with length >= 2 as bursts
        run_edges = np.diff(np.r_[0, in_burst.astype(int), 0])
        starts = np.flatnonzero(run_edges == +1)
        ends = np.flatnonzero(run_edges == -1) - 1
        bursts = []
        for s, e in zip(starts, ends):
            if (e - s + 1) >= 2:
                bursts.append(idx_sorted[s:e + 1])

        # Singles: not in burst AND both neighbors far
        singles_pos = np.where(~in_burst & (isi_prev > thr_single) & (isi_next > thr_single))[0]
        singles = idx_sorted[singles_pos]

        burst_spike_groups[cid] = bursts
        single_spike_indices[cid] = singles

        # ─────────── per-cluster JSON outputs ───────────
        _maybe_write([b.tolist() for b in bursts],
                     burst_type_path / f"burst_spike_groups_{cid}.json",
                     overwrite=overwrite)
        _maybe_write(singles.tolist(),
                     burst_type_path / f"single_spike_indices_{cid}.json",
                     overwrite=overwrite)

        # ─────────── DISTRIBUTION PLOTS (per cluster) ───────────
        # 1) Histogram of burst sizes (>=2)
        if bursts:
            sizes = np.array([len(b) for b in bursts], int)
            if sizes.size > 0:
                fig, ax = plt.subplots(figsize=(7, 5))
                bins = np.arange(2, sizes.max() + 2) - 0.5
                ax.hist(sizes, bins=bins)
                ax.set_xlabel("Spikes per burst")
                ax.set_ylabel("Count")
                ax.set_title(f"Cluster {cid} — Burst size distribution")
                ax.set_xticks(np.arange(2, sizes.max() + 1))
                _save_fig(fig, fig_dir / f"cluster_{cid}_burst_size_hist.png",
                          overwrite=overwrite)

        # 2) ISI distributions: all ISIs and within-burst ISIs
        if t_sorted.size > 1:
            all_isis = np.diff(t_sorted)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(all_isis, bins="auto")
            ax.set_xlabel("ISI (s)")
            ax.set_ylabel("Count")
            ax.set_title(f"Cluster {cid} — All ISIs")
            _save_fig(fig, fig_dir / f"cluster_{cid}_isi_hist_all.png",
                      overwrite=overwrite)

            if bursts:
                burst_isis = []
                for b in bursts:
                    tb = spike_times_sec[b]
                    if tb.size >= 2:
                        burst_isis.append(np.diff(tb))
                if burst_isis:
                    burst_isis = np.concatenate(burst_isis)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.hist(burst_isis, bins="auto")
                    ax.set_xlabel("Within-burst ISI (s)")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Cluster {cid} — Within-burst ISIs")
                    _save_fig(fig, fig_dir / f"cluster_{cid}_isi_hist_within_burst.png",
                              overwrite=overwrite)

        # 3) Amplitude distributions: singles vs all burst spikes
        if a_sorted is not None and a_sorted.size:
            a_singles = a_sorted[singles_pos] if singles_pos.size else np.array([])
            # burst spike positions in sorted subset
            burst_mask_sorted = np.zeros(idx_sorted.size, dtype=bool)
            for s, e in zip(starts, ends):
                if (e - s + 1) >= 2:
                    burst_mask_sorted[s:e+1] = True
            a_burst_all = a_sorted[burst_mask_sorted]

            if a_singles.size:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hist(a_singles, bins="auto")
                ax.set_xlabel("Amplitude (a.u.)")
                ax.set_ylabel("Count")
                ax.set_title(f"Cluster {cid} — SINGLE spike amplitudes")
                _save_fig(fig, fig_dir / f"cluster_{cid}_amp_hist_single.png",
                          overwrite=overwrite)

            if a_burst_all.size:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hist(a_burst_all, bins="auto")
                ax.set_xlabel("Amplitude (a.u.)")
                ax.set_ylabel("Count")
                ax.set_title(f"Cluster {cid} — BURST spike amplitudes (all sizes)")
                _save_fig(fig, fig_dir / f"cluster_{cid}_amp_hist_burst_all_sizes.png",
                          overwrite=overwrite)

            # 4) Amplitude by within-burst order — across ALL burst sizes
            if bursts:
                amp_by_order_all = {}
                for b in bursts:
                    for o, idx_spk in enumerate(b):
                        amp_by_order_all.setdefault(o, []).append(float(amplitudes[idx_spk]))
                if amp_by_order_all:
                    orders = sorted(amp_by_order_all.keys())
                    data = [np.asarray(amp_by_order_all[o], float) for o in orders]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.boxplot(data, showfliers=True, widths=0.6)
                    ax.set_xticks(np.arange(1, len(orders) + 1))
                    ax.set_xticklabels([f"Order {o+1}" for o in orders], rotation=0)
                    ax.set_ylabel("Amplitude (a.u.)")
                    ax.set_title(f"Cluster {cid} — Amplitude by within-burst order (all sizes)")
                    # NEW: overlay raw points
                    _overlay_scatter_on_box(ax, data_by_group=data,
                                            jitter=0.03, alpha=0.7, s=18,
                                            max_points=20000, seed=123)
                    _save_fig(fig, fig_dir / f"cluster_{cid}_amp_by_order_box_all_sizes.png",
                              overwrite=overwrite)

            # 5) Size-specific amplitude graphs with raw-point overlays
            if bursts:
                # Build collections
                size_to_all_amps = {}
                size_to_order_amps = {}
                for b in bursts:
                    k = len(b)
                    amps_b = amplitudes[b].astype(float)
                    size_to_all_amps.setdefault(k, []).append(amps_b)
                    d = size_to_order_amps.setdefault(k, {})
                    for o, idx_spk in enumerate(b):
                        d.setdefault(o, []).append(float(amplitudes[idx_spk]))

                for k in sorted(size_to_all_amps.keys()):
                    # (a) Histogram of all amps in bursts of size k
                    all_amps_k = np.concatenate(size_to_all_amps[k]) if size_to_all_amps[k] else np.array([])
                    if all_amps_k.size:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.hist(all_amps_k, bins="auto")
                        ax.set_xlabel("Amplitude (a.u.)")
                        ax.set_ylabel("Count")
                        ax.set_title(f"Cluster {cid} — Amplitudes for bursts of size {k}")
                        _save_fig(fig, fig_dir / f"cluster_{cid}_amp_hist_burstsize_{k}.png",
                                  overwrite=overwrite)

                    # (b) Boxplot of amplitude by order for size k + raw overlays
                    order_map = size_to_order_amps.get(k, {})
                    if order_map:
                        orders_k = sorted(order_map.keys())
                        data_k = [np.asarray(order_map[o], float) for o in orders_k]
                        if any(len(dk) for dk in data_k):
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.boxplot(data_k, showfliers=True, widths=0.6)
                            ax.set_xticks(np.arange(1, len(orders_k) + 1))
                            ax.set_xticklabels([f"Order {o+1}" for o in orders_k])
                            ax.set_ylabel("Amplitude (a.u.)")
                            ax.set_title(f"Cluster {cid} — Amplitude by order (burst size {k})")
                            # NEW: overlay raw points
                            _overlay_scatter_on_box(ax, data_by_group=data_k,
                                                    jitter=0.03, alpha=0.7, s=18,
                                                    max_points=20000, seed=123)
                            _save_fig(fig, fig_dir / f"cluster_{cid}_amp_by_order_box_burstsize_{k}.png",
                                      overwrite=overwrite)

    # ───────────────────────── summary DataFrame & CSV ─────────────────
    summary_rows = []

    for cid in clusters_of_interest:
        bursts = burst_spike_groups.get(cid, [])
        singles = single_spike_indices.get(cid, np.array([], int))

        lengths = np.array([len(b) for b in bursts], dtype=int) if bursts else np.array([], int)
        n_bursts = int(lengths.size)
        n_singles = int(singles.size)
        n_burst_spikes = int(lengths.sum()) if lengths.size else 0

        mean_amp = {}
        sem_amp = {}
        norm_ok = {}
        if amplitudes is not None and n_bursts > 0:
            amp_by_order = {}
            for b in bursts:
                for o, idx_spk in enumerate(b):
                    amp_by_order.setdefault(o, []).append(amplitudes[idx_spk])

            for o, vals in amp_by_order.items():
                vals = np.asarray(vals, float)
                mean_amp[o] = float(np.mean(vals)) if vals.size else np.nan
                sem_amp[o] = float(_safe_sem(vals))
                norm_ok[o] = bool(_safe_shapiro(vals)[0]) if vals.size >= 3 else False

            test_name, p_val, _post = run_stat_tests_for_bursts(bursts, amplitudes)
        else:
            test_name, p_val = None, np.nan

        row = dict(
            cluster_id=int(cid),
            n_single_spikes=n_singles,
            n_bursts=n_bursts,
            n_burst_spikes=n_burst_spikes,
            mean_spikes_per_burst=float(lengths.mean()) if n_bursts else np.nan,
            spike_amp_test=test_name,
            spike_amp_pvalue=float(p_val) if np.isfinite(p_val) else np.nan,
        )

        max_order = max(mean_amp.keys(), default=-1)
        for o in range(max_order + 1):
            row[f"Amplitude_Order_{o+1}"] = mean_amp.get(o, np.nan)
            row[f"SEM_Order_{o+1}"] = sem_amp.get(o, np.nan)
            row[f"Normal_Order_{o+1}"] = norm_ok.get(o, False)

        summary_rows.append(row)

    final_df = pd.DataFrame(summary_rows)

    core_cols = [
        "cluster_id", "n_single_spikes", "n_bursts", "n_burst_spikes",
        "mean_spikes_per_burst", "spike_amp_test", "spike_amp_pvalue"
    ]
    amp_cols = sorted([c for c in final_df.columns if c.startswith("Amplitude_Order_")],
                      key=lambda x: int(x.rsplit("_", 1)[-1]))
    sem_cols = sorted([c for c in final_df.columns if c.startswith("SEM_Order_")],
                      key=lambda x: int(x.rsplit("_", 1)[-1]))
    norm_cols = sorted([c for c in final_df.columns if c.startswith("Normal_Order_")],
                       key=lambda x: int(x.rsplit("_", 1)[-1]))

    for col in core_cols:
        if col not in final_df.columns:
            final_df[col] = np.nan

    final_df = final_df[core_cols + amp_cols + sem_cols + norm_cols]

    _maybe_write(final_df, burst_analysis_summary, overwrite=overwrite, index=False)

    return burst_spike_groups, single_spike_indices


def assign_cluster_types(py_ids, ni_ids, wi_ids):

    """
    Assigns types ('PY', 'NI', 'NW') to corresponding lists of cluster IDs within a dictionary.

    Args:
        cluster_dict (dict): Dictionary where keys are cluster IDs and values are the types.
        py_ids (list): List of cluster IDs to assign as 'PY'.
        ni_ids (list): List of cluster IDs to assign as 'NI'.
        nw_ids (list): List of cluster IDs to assign as 'WI'.

    Returns:
        dict: Modified dictionary with assigned cluster types.
    """
    cluster_types = {}

    for cid in py_ids:
        cluster_types[cid] = 'PY'
    for cid in ni_ids:
        cluster_types[cid] = 'NI'
    for cid in wi_ids:
        cluster_types[cid] = 'WI'

    return cluster_types

def main_phase_analysis(
        kilosort_sorter_output_folder,
        lfp_binary_file,
        burst_spike_groups,
        single_spike_indices,
        selected_shifted_events,
        cluster_types,
        custom_analytics_folder,
        n_channels,
        lower_limit,
        upper_limit,
        data_file_type,
        bit_to_uv_val,
        units_in,
        *,                      # everything below is keyword-only
        overwrite=False,        # ← NEW: default keeps existing files
        lfp_sampling_rate,      # REQUIRED: sampling rate of lfp_binary_file in Hz (e.g. 1250)
        spike_lfp_index=None,   # optional: A1's spike_lfp_index.npy (one LFP sample per Kilosort spike)
        spike_near_gap=None,    # A1's spike_near_gap.npy; required with spike_lfp_index
        peak_channels=None,     # {cluster_id: channel index in the LFP file}; e.g. from the C1 analyzer
):
    """
    Gamma-band spike-LFP phase analysis with overwrite protection.

    Spikes (Kilosort samples, fs from ops) and the LFP (lfp_sampling_rate) are
    separate clocks. Each spike is mapped to its nearest LFP sample, either from
    `spike_lfp_index` (A1's alignment output) or as round(sample * lfp_fs / fs).
    The LFP is filtered at lfp_sampling_rate, which must be given explicitly. Burst /
    single spike indices are positions in Kilosort's spike list and are mapped to LFP
    samples the same way. Spikes flagged in spike_near_gap (LFP near lost samples)
    are excluded from every statistic: all spikes, bursts and singles.

    Peak channel per cluster: pass `peak_channels` (recommended, from the current
    analyzer: si.get_template_extremum_channel(analyzer, outputs="index")). Otherwise
    each cluster's channel is the best channel of the Kilosort template most of its
    spikes came from (spike_templates.npy), which also works for Phy merges / splits
    whose new cluster IDs have no template row of their own.

    Returns
    -------
    list[pathlib.Path]
        Absolute paths of every CSV / PNG that was *actually* written.
        Handy for showing inside a Voilà log panel.
    """
    # ───────────────────────────── imports & helpers ──────────────────────────
    import os, copy, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from pathlib import Path
    from numpy.random import default_rng
    from mne.filter import filter_data, create_filter
    from scipy.signal import hilbert
    from scipy.stats  import circmean, circstd
    from pycircstat.tests import rayleigh
    from kilosort.io import load_ops
    from kilosort.data_tools import get_best_channels

    # ------------- overwrite-aware writer ------------------------------------
    file_log = []                                               # paths written

    def _maybe_write(obj, dst: Path, *, overwrite: bool, **kw):
        """
        Robust “write-if-needed” helper used throughout the analysis.

        Parameters
        ----------
        obj : object
            • matplotlib Figure       → detected via ``hasattr(obj, "savefig")``
            • pandas DataFrame        → detected via ``hasattr(obj, "to_csv")``
            • anything else           → written as plain text (``str(obj)``)

        dst : pathlib.Path
            Destination filename.  Parent folders are created automatically.

        overwrite : bool
            If *False* and *dst* already exists → **nothing is written**.

        **kw
            Forwarded verbatim to ``savefig`` or ``to_csv`` (e.g. ``dpi=300``,
            ``index=False`` …).

        Side effects
        ------------
        • Prints a short status message (“Saved → …”, “Over-written → …” or
          “Skipped (exists, overwrite=False): …”) so the Voilà log panel shows
          progress in real time.

        • Appends *dst* to the outer‐scope list ``file_log`` **only when** the
          file is actually written.  This lets the caller display a neat “Files
          written:” summary at the end of the run.
        """
        dst = Path(dst)
        pre_exists = dst.exists()  # capture *before* any writing

        # ─── early-exit if we are not allowed to overwrite ─────────────────
        if pre_exists and not overwrite:
            print(f"Skipped (exists, overwrite=False): {dst}")
            return

        # make sure the target directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        # ─── dispatch by object type ───────────────────────────────────────
        if hasattr(obj, "savefig"):  # matplotlib Figure
            obj.savefig(dst, **kw)
        elif hasattr(obj, "to_csv"):  # pandas / polars DataFrame
            obj.to_csv(dst, **kw)
        else:  # generic: write str(obj)
            dst.write_text(str(obj))

        # ─── bookkeeping / log message ─────────────────────────────────────
        verb = "Over-written" if pre_exists and overwrite else "Saved"
        print(f"{verb} → {dst}")
        file_log.append(dst)  # <- record that we actually wrote it

    # ─────────────────────  ORIGINAL ANALYSIS CODE  ───────────────────────────
    #       ░░░  NOTHING BELOW THIS BOX WAS CHANGED EXCEPT FILE SAVES  ░░░
    # -------------------------------------------------------------------------

    # ---------- 0) load core arrays ------------------------------------------
    results_dir = Path(kilosort_sorter_output_folder)
    ops         = load_ops(results_dir / "ops.npy")
    fs          = ops["fs"]

    spike_times        = np.load(results_dir / "spike_times.npy")
    spike_clusters     = np.load(results_dir / "spike_clusters.npy")
    amplitudes         = np.load(results_dir / "amplitudes.npy")
    spike_times_sec    = spike_times / fs
    orig_idx           = np.arange(len(spike_times))

    # ---------- 1) find best LFP channel per cluster --------------------------
    if peak_channels is not None:
        missing = [c for c in units_in if c not in peak_channels]
        if missing:
            raise ValueError(f"peak_channels has no entry for clusters {missing[:10]}")
        cid2chan = {c: int(peak_channels[c]) for c in units_in}
    else:
        # get_best_channels() is per TEMPLATE row, not per cluster ID: go through each
        # cluster's dominant template (cluster IDs from Phy merges/splits exceed the rows)
        template_best = np.asarray(get_best_channels(results_dir))
        spike_templates = np.load(results_dir / "spike_templates.npy").ravel()
        clusters_all = spike_clusters.ravel()
        cid2chan = {}
        for c in units_in:
            tmpl = spike_templates[clusters_all == c]
            if tmpl.size == 0:
                raise ValueError(f"Cluster {c} has no spikes in spike_clusters.npy.")
            dominant = int(np.bincount(tmpl.astype(np.int64)).argmax())
            if dominant >= template_best.size:
                raise ValueError(f"Template {dominant} of cluster {c} has no best-channel entry.")
            cid2chan[c] = int(template_best[dominant])

    # ---------- 2) memory-map the LFP binary ----------------------------------
    dtype            = np.dtype(data_file_type)
    bytes_per_sample = dtype.itemsize * n_channels
    n_samples        = Path(lfp_binary_file).stat().st_size // bytes_per_sample
    mm               = np.memmap(lfp_binary_file, dtype=dtype, mode='r',
                                 shape=(n_samples, n_channels))
    # keep only the columns we need
    chan_cols        = [cid2chan[c] for c in units_in]
    slice_idx        = {ch: i for i, ch in enumerate(chan_cols)}
    cid2slice        = {cid: slice_idx[cid2chan[cid]] for cid in units_in}
    lfp_arr          = mm[:, chan_cols] * bit_to_uv_val   # convert to µV

    # ---------- 2b) spike -> LFP sample mapping (separate clocks) -------------
    if lfp_sampling_rate is None or not np.isfinite(lfp_sampling_rate) or lfp_sampling_rate <= 0:
        raise ValueError("lfp_sampling_rate is required: the sampling rate of lfp_binary_file in Hz (e.g. 1250).")
    lfp_fs = float(lfp_sampling_rate)
    expected_idx = np.rint(spike_times.astype(np.float64).ravel() * lfp_fs / fs).astype(np.int64)
    if spike_lfp_index is not None:
        spike_lfp_idx = np.asarray(spike_lfp_index, dtype=np.int64).ravel()
        if spike_lfp_idx.size != spike_times.size:
            raise ValueError(f"spike_lfp_index has {spike_lfp_idx.size} entries for {spike_times.size} spikes; "
                             "it must be A1's output for this exact Kilosort run.")
        off = np.abs(spike_lfp_idx - expected_idx)
        if off.size and off.max() > 1:
            raise ValueError(
                f"spike_lfp_index disagrees with lfp_sampling_rate={lfp_fs:g} Hz by up to {int(off.max())} "
                "LFP samples. The LFP rate is wrong, or the index comes from another LFP / sort."
            )
        if spike_near_gap is None:
            raise ValueError("Pass spike_near_gap (A1's spike_near_gap.npy) together with spike_lfp_index.")
    else:
        spike_lfp_idx = expected_idx
    if spike_near_gap is None:
        near_gap = np.zeros(spike_times.size, dtype=bool)
    else:
        near_gap = np.asarray(spike_near_gap, dtype=bool).ravel()
        if near_gap.size != spike_times.size:
            raise ValueError(f"spike_near_gap has {near_gap.size} entries for {spike_times.size} spikes.")
    if spike_lfp_idx.size and spike_lfp_idx.max() >= n_samples:
        raise ValueError(
            f"Spikes map past the end of the LFP ({spike_lfp_idx.max():,} >= {n_samples:,} samples). "
            "If the LFP is downsampled (e.g. 1250 Hz), pass lfp_sampling_rate=1250 "
            "or spike_lfp_index from A1."
        )

    # ---------- 3) pre-compute γ-phase per channel ----------------------------
    def bandpass(x, lo, hi, sf):
        return filter_data(x.astype(np.float64), sfreq=sf,
                           l_freq=lo, h_freq=hi,
                           method="fir", phase="zero")

    def gamma_phase(trace):
        filt = bandpass(trace, lower_limit, upper_limit, lfp_fs)   # LFP rate, not spike rate
        return np.mod(np.degrees(np.angle(hilbert(filt))), 360)  # 0–360°

    phase_cache = {}
    for cid in units_in:
        phase_cache[cid] = gamma_phase(lfp_arr[:, cid2slice[cid]])

    # ---------- 4) masks excluding edge / opto artefacts ----------------------
    max_fir = create_filter(None, lfp_fs, lower_limit, upper_limit,
                            method="fir", phase="zero").size      # length in LFP samples
    edge_ok = (spike_lfp_idx >= max_fir) & (spike_lfp_idx < n_samples - max_fir)

    ev_sec   = np.asarray(selected_shifted_events, float)
    if ev_sec.size:
        ix_prev  = np.searchsorted(ev_sec, spike_times_sec, "right") - 1
        prev_ev  = np.where(ix_prev >= 0, ev_sec[np.clip(ix_prev, 0, None)], -np.inf)
        t_since  = spike_times_sec - prev_ev
        non_opto = t_since > 0.7
    else:
        non_opto = np.ones(spike_times_sec.shape, dtype=bool)   # no events: nothing to exclude

    valid    = edge_ok & non_opto & ~near_gap         # gap-adjacent spikes out everywhere
    keep_idx = orig_idx[valid]               # indices into filtered arrays

    # filtered arrays
    spike_times        = spike_times[valid]
    spike_lfp_idx      = spike_lfp_idx[valid]
    spike_times_sec    = spike_times_sec[valid]
    spike_clusters     = spike_clusters[valid]
    amplitudes         = amplitudes[valid]

    # map “old → new” spike indices (for burst/single look-ups)
    old2new = np.full(orig_idx.size, -1, int)
    old2new[keep_idx] = np.arange(keep_idx.size)

    # ---------- 5) per-cluster phase statistics ------------------------------
    rng              = default_rng(42)
    stats_rows       = []
    per_group_phases = {"NI": [], "WI": [], "PY": []}

    for cid in units_in:

        # ---- extract spike indices for this cluster -------------------------
        c_mask  = spike_clusters == cid
        if not c_mask.any():
            print(f"cid {cid}: no valid spikes after masks")
            continue

        ch_phase = phase_cache[cid]
        phases   = ch_phase[spike_lfp_idx[c_mask]]   # LFP sample of each spike
        phases   = phases[~np.isnan(phases)]

        pref_phi = circmean(phases, high=360, low=0)
        gtype    = cluster_types.get(cid, "UNK")
        if gtype in per_group_phases:
            per_group_phases[gtype].append(pref_phi)

        # ---- bursts / singles ----------------------------------------------
        raw_bursts  = burst_spike_groups.get(cid, [])
        bursts_ok   = [grp for grp in raw_bursts
                       if np.all(old2new[grp] >= 0)]
        first_idx   = np.array([old2new[grp[0]] for grp in bursts_ok], int)
        second_idx  = np.array([old2new[grp[1]] for grp in bursts_ok
                                                if len(grp) > 1], int)
        singles_ok  = old2new[single_spike_indices.get(cid,
                                                       np.array([], int))]
        singles_ok  = singles_ok[singles_ok >= 0]

        # first_idx / second_idx / singles_ok are positions in the kept spike list:
        # map them to those spikes' LFP samples before reading the phase
        first_phi   = ch_phase[spike_lfp_idx[first_idx]]
        second_phi  = ch_phase[spike_lfp_idx[second_idx]]
        single_phi  = ch_phase[spike_lfp_idx[singles_ok]]
        first_phi   = first_phi[~np.isnan(first_phi)]
        second_phi  = second_phi[~np.isnan(second_phi)]
        single_phi  = single_phi[~np.isnan(single_phi)]

        # ---- store per-cell summary row ------------------------------------
        row = dict(
            cluster_id      = cid,
            cell_type       = gtype,
            pref_phase_deg  = pref_phi,
            n_first         = first_phi.size,
            n_second        = second_phi.size,
            n_single        = single_phi.size,
            mean_first_deg  = circmean(first_phi,  high=360, low=0) if first_phi.size else np.nan,
            mean_second_deg = circmean(second_phi, high=360, low=0) if second_phi.size else np.nan,
            mean_single_deg = circmean(single_phi, high=360, low=0) if single_phi.size else np.nan,
            sem_first_deg   = circstd(first_phi,  high=360, low=0)/np.sqrt(first_phi.size)  if first_phi.size>1 else np.nan,
            sem_second_deg  = circstd(second_phi, high=360, low=0)/np.sqrt(second_phi.size) if second_phi.size>1 else np.nan,
            sem_single_deg  = circstd(single_phi, high=360, low=0)/np.sqrt(single_phi.size) if single_phi.size>1 else np.nan,
        )
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)

    # ---------- 6) save per-cell statistics ----------------------------------
    stats_csv = Path(custom_analytics_folder) / "spike_phase_analysis_1.csv"
    _maybe_write(stats_df, stats_csv, overwrite=overwrite, index=False)

    # ---------- 7) population phase-density plots ----------------------------
    bins     = np.linspace(0, 360, 25)                    # 24 bins of 15°, last edge 360°
    centers  = (bins[:-1] + bins[1:])/2
    colors   = dict(NI="tab:blue", WI="tab:green", PY="tab:red")

    figs = {}
    for grp, col in colors.items():
        if not per_group_phases[grp]:
            continue
        fig, ax = plt.subplots(figsize=(5,4))
        ph_arr  = np.hstack(per_group_phases[grp])
        h, _    = np.histogram(ph_arr, bins=bins)
        if h.sum() == 0:
            plt.close(fig)
            continue
        h       = h / h.sum()                              # normalised
        ax.bar(centers, h, width=15, color=col, edgecolor="k")
        ax.set(title=f"{grp} preferred phase", xlabel="Phase (°)",
               ylabel="Density", xlim=(0,360))
        figs[grp] = fig

    # save figures
    fig_dir = Path(custom_analytics_folder) / "phase_density_figs"
    for grp, fig in figs.items():
        fp = fig_dir / f"{grp}_phase_density.png"
        _maybe_write(fig, fp, overwrite=overwrite, dpi=300)
        plt.close(fig)

    # (add any additional _maybe_write calls you need)

    # -------------------------------------------------------------------------
    # 8) Return list of real outputs
    # -------------------------------------------------------------------------
    return file_log

# (A second, duplicate load_ops() used to be here. It silently replaced the
#  type-hinted version above, so it was removed. Use load_ops() defined near the top.)

##New optogenetics screen 473 nm laser
#############
##########
###############
##############
##############
import numpy as np


# ============================================================
# BASIC UTILITIES
# ============================================================
def mad_sigma(x):
    """Robust estimate of standard deviation using MAD."""
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def moving_average(x, win):
    """Simple moving average smoothing."""
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def hysteresis_segments(x, th_low, th_high):
    """
    Detect segments using hysteresis thresholding.
    """
    segs = []
    state = 0
    start = None

    for i, v in enumerate(x):
        if state == 0:
            if v >= th_high:
                state = 1
                start = i
        else:
            if v <= th_low:
                state = 0
                segs.append((start, i))

    return segs


# ============================================================
# MAIN FUNCTION: AI14 PULSE DETECTION
# ============================================================


def detect_laser_pulses(
    rec_nidaq=None,
    ai_channel="AI15",
    smooth_ms=0.1,
    min_width_ms=0.8,
    max_width_ms=None,
    high_thresh_sigma=3.5,
    low_thresh_sigma=2.0,
    min_peak_sigma=10,
    return_debug=False,
    return_segment_info=False,
):
    """
    Detect pulse-like segments from the selected analog-input channel.

    Detection occurs entirely in the NI-DAQ timeline. This function
    does not use a protocol schedule, assign experimental phases, or
    map events into Neuropixels frames.

    Parameters
    ----------
    min_width_ms : float
        Minimum accepted segment width.
    max_width_ms : float or None
        Maximum accepted segment width. Leave as None to preserve
        the original minimum-width-only behavior.
    return_debug : bool
        Return raw and filtered signals plus segment information.
        This retains substantial data in memory.
    return_segment_info : bool
        Return compact onset, offset, width and peak information
        without retaining the raw and filtered signals.

    Returns
    -------
    frames : np.ndarray
        Segment-onset indices in the NI-DAQ recording.
    times : np.ndarray
        Segment-onset times relative to NI-DAQ recording start.
    debug or segment_info : dict, optional
        Returned when the corresponding option is enabled.
    """

    print("\n[STEP] Detecting Laser AIC pulses...")

    # ================= LOAD SIGNAL =================
    if rec_nidaq is None:
        raise ValueError(
            "rec_nidaq must be provided."
        )

    rec_ch = rec_nidaq.select_channels([
        ai_channel
    ])

    sr = float(
        rec_nidaq.get_sampling_frequency()
    )

    y = (
        rec_ch
        .get_traces()
        .flatten()
    )

    t = np.asarray(
        rec_nidaq.get_times(),
        dtype=float,
    )

    if len(y) == 0 or len(t) == 0:
        raise RuntimeError(
            f"{ai_channel} recording is empty."
        )

    if len(y) != len(t):
        raise RuntimeError(
            "AI-channel trace and timestamp lengths differ."
        )

    t = t - t[0]

    print(f"[INFO] Sampling rate: {sr} Hz")
    print(f"[INFO] Total samples: {len(y)}")

    # ================= SMOOTH =================
    win = max(
        1,
        int(
            round(
                smooth_ms
                * sr
                / 1000.0
            )
        ),
    )

    y_f = moving_average(
        y,
        win,
    )

    # ================= THRESHOLD =================
    med = float(
        np.median(y_f)
    )

    sigma = float(
        mad_sigma(y_f)
    )

    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError(
            "Could not estimate positive AI-channel noise using MAD."
        )

    th_high = (
        med
        + high_thresh_sigma * sigma
    )

    th_low = (
        med
        + low_thresh_sigma * sigma
    )

    if th_low >= th_high:
        raise ValueError(
            "low_thresh_sigma must be smaller than "
            "high_thresh_sigma."
        )

    print(
        f"[INFO] median={med:.3f}, "
        f"sigma={sigma:.3f}"
    )

    print(
        f"[INFO] thresholds: "
        f"low={th_low:.3f}, "
        f"high={th_high:.3f}"
    )

    # ================= SEGMENT DETECTION =================
    segs_raw = hysteresis_segments(
        y_f,
        th_low,
        th_high,
    )

    print(
        f"[INFO] Raw segments: "
        f"{len(segs_raw)}"
    )

    # ================= WIDTH FILTER =================
    minN = max(
        1,
        int(
            round(
                min_width_ms
                * sr
                / 1000.0
            )
        ),
    )

    if max_width_ms is None:
        segs = [
            (int(s), int(e))
            for s, e in segs_raw
            if (e - s) >= minN
        ]

    else:
        maxN = max(
            1,
            int(
                round(
                    max_width_ms
                    * sr
                    / 1000.0
                )
            ),
        )

        if maxN < minN:
            raise ValueError(
                "max_width_ms must be greater than or equal "
                "to min_width_ms."
            )

        segs = [
            (int(s), int(e))
            for s, e in segs_raw
            if minN <= (e - s) <= maxN
        ]

    print(
        f"[INFO] After width filter: "
        f"{len(segs)}"
    )

    # ================= PEAK FILTER =================
    real_segs = []
    real_peak_values = []
    real_peak_above_median = []

    for s, e in segs:
        if (
            s < 0
            or e <= s
            or e >= len(y_f)
            or e >= len(t)
        ):
            continue

        segment = y_f[s:e]

        if len(segment) == 0:
            continue

        peak_value = float(
            np.max(
                np.abs(segment)
            )
        )

        peak_above_median = float(
            np.max(
                segment - med
            )
        )

        if (
            peak_value
            > min_peak_sigma * sigma
        ):
            real_segs.append(
                (s, e)
            )

            real_peak_values.append(
                peak_value
            )

            real_peak_above_median.append(
                peak_above_median
            )

    print(
        f"[INFO] Final pulses: "
        f"{len(real_segs)}"
    )

    # ================= COMPACT OUTPUT =================
    onset_frames = np.asarray(
        [s for s, _ in real_segs],
        dtype=np.int64,
    )

    offset_frames = np.asarray(
        [e for _, e in real_segs],
        dtype=np.int64,
    )

    segment_width_samples = (
        offset_frames
        - onset_frames
    )

    segment_width_ms = (
        segment_width_samples
        / sr
        * 1000.0
    )

    peak_values = np.asarray(
        real_peak_values,
        dtype=float,
    )

    peak_above_median = np.asarray(
        real_peak_above_median,
        dtype=float,
    )

    onset_times = t[
        onset_frames
    ]

    offset_times = t[
        offset_frames
    ]

    segment_info = {
        "onset_frames": onset_frames,
        "offset_frames": offset_frames,
        "onset_times": onset_times,
        "offset_times": offset_times,
        "segment_width_samples": segment_width_samples,
        "segment_width_ms": segment_width_ms,
        "peak_values": peak_values,
        "peak_above_median": peak_above_median,
        "sampling_rate": sr,
        "median": med,
        "sigma": sigma,
        "threshold_high": th_high,
        "threshold_low": th_low,
    }

    frames = onset_frames
    times = onset_times

    # ================= OPTIONAL OUTPUTS =================
    if return_debug:
        debug = {
            "raw_signal": y,
            "filtered_signal": y_f,
            "median": med,
            "sigma": sigma,
            "threshold_high": th_high,
            "threshold_low": th_low,
            "segments_raw": segs_raw,
            "segments_width_filtered": segs,
            "segments_final": real_segs,
            "segment_info": segment_info,
        }

        return frames, times, debug

    if return_segment_info:
        return frames, times, segment_info

    return frames, times

# ============================================================
# OPTIONAL: CLEAN DOUBLE DETECTIONS
# ============================================================
def enforce_refractory(frames, sampling_rate, min_interval_ms=1.0):
    """
    Remove detections that are too close together.

    Args:
        frames: detected indices
        sampling_rate: Hz
        min_interval_ms: minimum allowed interval

    Returns:
        cleaned frames
    """
    min_interval = min_interval_ms / 1000

    clean = [frames[0]]

    for f in frames[1:]:
        if (f - clean[-1]) / sampling_rate > min_interval:
            clean.append(f)

    return np.array(clean)


# ============================================================
# OPTIONAL: AMPLITUDE EXTRACTION
# ============================================================
def extract_pulse_amplitudes(rec_nidaq, frames, ai_channel="AI14", window=50):
    """
    Extract peak amplitude for each pulse.
    """
    rec_ch = rec_nidaq.select_channels([ai_channel])
    y = rec_ch.get_traces().flatten()

    amps = []
    for f in frames:
        segment = y[f:f+window]
        amps.append(np.max(segment))

    return np.array(amps)


# ============================================================
# OPTIONAL: BLOCK SEGMENTATION
# ============================================================
def segment_blocks(times, gap_threshold=3.0):
    """
    Segment pulses into blocks based on time gaps.
    """
    block_ids = np.zeros(len(times), dtype=int)
    block = 0

    for i in range(1, len(times)):
        if times[i] - times[i-1] > gap_threshold:
            block += 1
        block_ids[i] = block

    return block_ids


# ============================================================
# PREPROCESSING SAFETY HELPERS AND UNIT SUMMARY (added 2026-10-21)
# Settings checks on saved folders, cropping events to the recording,
# the timestamp gap check, and readers that pick up Kilosort, UnitRefine,
# Bombcell and CellExplorer results if they exist.
# ============================================================

SETTINGS_STAMP_NAME = "nam_preprocessing_settings.json"

# Largest backward timestamp step accepted as an Open Ephys synchronization correction (s).
# Streams synchronized to another (main) stream get small jumps whenever the synchronizer
# updates its mapping: e.g. up to 11 us in a 2-min NI-DAQ stream, and sync pulses off by up
# to ~70 us from the main stream in a 2-h OneBox probe stream (2026-09-06). Larger backward
# steps are treated as clock faults. Adjust here if a recording needs it.
SYNC_CORRECTION_MAX_S = 0.25e-3

# Settings shared by the spike and LFP branches (made in notebook cell 1.1).
# S3 stores exactly these next to the trimmed timestamps; A1 compares them across outputs.
SHARED_SETTINGS_KEYS = (
    "recording_root", "stream_name", "remove_first_minutes", "probe_crop_start_frame",
    "bad_channels_removed", "phase_shift", "n_channels", "raw_sampling_rate_hz", "raw_n_frames",
)


def _jsonable(obj):
    """Convert numpy / Path values so settings can be written to and compared as JSON."""
    import numpy as np
    from pathlib import Path
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in list(obj)]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and obj.is_integer():
        return obj
    return obj


def write_settings_stamp(folder, settings, stamp_name=SETTINGS_STAMP_NAME):
    """Save the settings used to make a saved recording folder (or file) next to it."""
    import json
    from pathlib import Path
    folder = Path(folder)
    stamp = folder / stamp_name if folder.is_dir() else folder.with_name(folder.name + ".settings.json")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(_jsonable(settings), indent=2, sort_keys=True))
    return stamp


def check_settings_stamp(folder, settings, stamp_name=SETTINGS_STAMP_NAME):
    """
    Compare the current settings with the ones saved next to `folder`.

    Returns
    -------
    "match"     settings are identical
    "no_stamp"  the folder exists but has no settings file (made before this check existed)
    Raises ValueError listing every setting that differs.
    """
    import json
    from pathlib import Path
    folder = Path(folder)
    stamp = folder / stamp_name if folder.is_dir() else folder.with_name(folder.name + ".settings.json")
    if not stamp.exists():
        return "no_stamp"
    saved = json.loads(stamp.read_text())
    current = json.loads(json.dumps(_jsonable(settings), sort_keys=True))
    diffs = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key, "<missing>") != current.get(key, "<missing>"):
            s, c = saved.get(key, "<missing>"), current.get(key, "<missing>")
            if isinstance(s, list) and len(s) > 8:
                s = f"list of {len(s)}"
            if isinstance(c, list) and len(c) > 8:
                c = f"list of {len(c)}"
            diffs.append(f"  {key}: saved = {s!r}, current = {c!r}")
    if diffs:
        raise ValueError(
            f"Existing output was made with different settings:\n{folder}\n"
            + "\n".join(diffs)
            + "\nMove or delete it, then rerun this cell."
        )
    return "match"


def crop_events_to_recording(shifted_events, duration_s):
    """
    Keep only events inside the cropped recording.

    shifted_events : event times in seconds from the first kept sample
                     (output of process_block on the cropped recording).
    duration_s     : length of the cropped recording in seconds.

    Returns (kept_times, kept_original_indices). Times are on the same zero as
    Kilosort spike_times / sampling_rate.
    """
    import numpy as np
    ev = np.asarray(shifted_events, dtype=float).ravel()
    keep = (ev >= 0) & (ev < float(duration_s))
    return ev[keep], np.flatnonzero(keep)


def check_timestamp_gaps(timestamps, sample_numbers, fs, tolerance_samples=0.5, max_report=20):
    """
    Look for dropped, repeated or out-of-order samples in a trimmed Open Ephys stream.

    sample_numbers steps are classified separately:
        forward gaps   step > 1   (missing_samples_total = sum of step - 1)
        repeats        step == 0
        backward jumps step < 0
    Timestamps are checked for non-finite values (NaN / inf) and for steps more than
    `tolerance_samples` away from 1/fs (or 1, if timestamps are stored as sample counts).

    Returns a dict (JSON-safe). ok is True only if none of the above occur.
    Gaps do not change Kilosort's sample indices, but they mean
    "sample index / fs" is no longer exact acquisition time; use the trimmed
    timestamps for that instead.
    """
    import numpy as np
    sn = np.asarray(sample_numbers)
    ts = np.asarray(timestamps, dtype=np.float64)
    report = {"n_samples": int(sn.size), "fs_hz": float(fs)}
    if sn.size != ts.size:
        raise ValueError(f"timestamps ({ts.size}) and sample_numbers ({sn.size}) differ in length.")

    finite = np.isfinite(ts)
    n_nonfinite = int(np.count_nonzero(~finite))
    report["n_nonfinite_timestamps"] = n_nonfinite
    report["first_nonfinite_timestamps"] = [int(i) for i in np.flatnonzero(~finite)[:max_report]]

    if sn.size < 2:
        raise ValueError(f"The clock check needs at least 2 samples; got {sn.size}.")

    d_sn = np.diff(sn.astype(np.int64))
    forward = np.flatnonzero(d_sn > 1)
    repeats = np.flatnonzero(d_sn == 0)
    backward = np.flatnonzero(d_sn < 0)
    bad_sn = np.flatnonzero(d_sn != 1)

    d_ts = np.diff(ts)
    d_ts_ok = np.isfinite(d_ts)
    median_step = float(np.median(d_ts[d_ts_ok])) if d_ts_ok.any() else float("nan")
    # Older Open Ephys versions store sample counts (step 1) instead of seconds
    ts_are_counts = bool(np.isfinite(median_step) and abs(median_step - 1.0) < 1e-6)
    expected = 1.0 if ts_are_counts else 1.0 / float(fs)
    report["timestamps_are_sample_counts"] = ts_are_counts
    # Steps touching a non-finite value are reported via n_nonfinite_timestamps
    irregular = d_ts_ok & (np.abs(d_ts - expected) > tolerance_samples * expected)
    # Open Ephys synchronization corrections: continuous sample numbers, timestamp step off the
    # nominal by less than SYNC_CORRECTION_MAX_S (not repeated). Reported, not counted as gaps.
    corrections = (irregular & (d_sn == 1) & (np.abs(d_ts - expected) < SYNC_CORRECTION_MAX_S) & (d_ts != 0)
                   if not ts_are_counts else np.zeros_like(irregular))
    bad_ts = np.flatnonzero(irregular & ~corrections)
    report["n_timestamp_backward_steps"] = int(np.count_nonzero(d_ts_ok & (d_ts <= 0) & ~corrections))
    # Timestamp jumps the sample numbers don't explain: with continuous sample numbers any step
    # off the nominal beyond a synchronization correction; across a sample-number gap a step
    # that differs from (missing samples + 1) / fs by more than half a sample plus the
    # correction limit. Either direction. Real dropped samples (timestamps jump with the
    # sample numbers) stay allowed.
    if ts_are_counts:
        expected_gap = d_sn.astype(np.float64)
        tol_gap = 0.5
    else:
        expected_gap = d_sn.astype(np.float64) / float(fs)
        tol_gap = 0.5 / float(fs) + SYNC_CORRECTION_MAX_S
    unexplained = d_ts_ok & (d_sn >= 1) & (((d_sn == 1) & irregular & ~corrections)
                                           | ((d_sn > 1) & (np.abs(d_ts - expected_gap) > tol_gap)))
    report["n_unexplained_timestamp_jumps"] = int(np.count_nonzero(unexplained))
    report["first_unexplained_timestamp_jumps"] = [
        {"index": int(i), "step_s": float(d_ts[i]), "sample_number_step": int(d_sn[i])}
        for i in np.flatnonzero(unexplained)[:max_report]]
    n_corr = int(np.count_nonzero(corrections))
    report["n_sync_corrections"] = n_corr
    report["largest_sync_correction_s"] = float(np.max(np.abs(d_ts[corrections] - expected))) if n_corr else 0.0
    report["n_backward_sync_corrections"] = int(np.count_nonzero(corrections & (d_ts < 0)))
    report["sync_correction_limit_s"] = SYNC_CORRECTION_MAX_S

    report.update(
        n_sample_number_gaps=int(bad_sn.size),
        n_forward_gaps=int(forward.size),
        missing_samples_total=int(np.sum(d_sn[forward] - 1)) if forward.size else 0,
        n_repeated_samples=int(repeats.size),
        n_backward_jumps=int(backward.size),
        first_sample_number_gaps=[
            {"index": int(i), "step": int(d_sn[i])} for i in bad_sn[:max_report]
        ],
        n_timestamp_gaps=int(bad_ts.size),
        first_timestamp_gaps=[
            {"index": int(i), "step_s": float(d_ts[i])} for i in bad_ts[:max_report]
        ],
        median_timestamp_step_s=median_step,
        implied_fs_hz=float(fs) if ts_are_counts else (
            float(1.0 / median_step) if median_step > 0 else float("nan")),
        ok=bool(bad_sn.size == 0 and bad_ts.size == 0 and n_nonfinite == 0),
    )
    # Usable for sorting / timing: forward gaps (lost samples) are allowed and reported; NaN/inf
    # timestamps, repeated or backward sample numbers, backward timestamp steps beyond the
    # synchronization-correction limit and timestamp jumps the sample numbers don't explain are not.
    report["clock_ok_for_sorting"] = bool(n_nonfinite == 0 and repeats.size == 0 and backward.size == 0
                                          and report["n_timestamp_backward_steps"] == 0
                                          and report["n_unexplained_timestamp_jumps"] == 0)
    return report


def events_to_probe_samples(event_times, probe_timestamps, fs, first_sample=0, extrapolate=True,
                            return_sides=False, sample_numbers=None, report=None):
    """
    Map event times onto the probe's sample index, using the probe's own timestamps.

    event_times     : event times in the same units Neo gives for this session
                      (Open Ephys >= 0.6: synchronized seconds; older: counts / fs).
    probe_timestamps: the probe stream's FULL timestamps.npy (seconds, or sample counts
                      in older Open Ephys; converted here).
    first_sample    : use only probe_timestamps[first_sample:] (e.g. the retained part
                      after a crop, when the discarded part has invalid timestamps).
                      Events before that sample are then unmappable: position NaN.
    Returns (sample_position, in_gap, outside):
        sample_position : float sample index in the full probe stream (fractional
                          between samples). Outside the recording: extrapolated at fs if
                          extrapolate=True (older behaviour), NaN if extrapolate=False.
        in_gap          : True if the event falls inside a forward gap (lost samples)
        outside         : True if the event is before the first or after the last sample
    With return_sides=True, `outside` is replaced by two arrays (before, after).

    Spike times / fs and these positions / fs are both on the probe's sample clock, so
    they stay aligned even across dropped samples, where "time - start" would not.

    sample_numbers : the probe's sample_numbers.npy (same length); if given, gaps are taken
                     from it (step > 1), otherwise from timestamp steps > 1.5 / fs.
    report         : dict, filled with the synchronization corrections found (see
                     timestamps_to_seconds). With corrections (small backward steps), positions
                     are interpolated on the running maximum of the timestamps, so an event
                     within a correction can be off by up to the correction size.
    """
    import numpy as np
    first_sample = int(first_sample)
    rep = {} if report is None else report
    ts, _ = timestamps_to_seconds(np.asarray(probe_timestamps)[first_sample:], fs, report=rep)
    if rep.get("n_backward_corrections"):
        ts = np.maximum.accumulate(ts)          # new array (the input may be a read-only memmap)
    ev = np.asarray(event_times, dtype=np.float64).ravel()
    idx = np.arange(ts.size, dtype=np.float64)
    pos = np.interp(ev, ts, idx)
    before, after = ev < ts[0], ev > ts[-1]
    if extrapolate:
        pos[before] = (ev[before] - ts[0]) * fs if first_sample == 0 else np.nan
        pos[after] = (ts.size - 1) + (ev[after] - ts[-1]) * fs
    else:
        pos[before | after] = np.nan
    if sample_numbers is not None:
        sn = np.asarray(sample_numbers)[first_sample:]
        if sn.size != ts.size:
            raise ValueError(f"sample_numbers ({sn.size}) and timestamps ({ts.size}) differ in length.")
        gap_after = np.flatnonzero(np.diff(sn.astype(np.int64)) > 1)
        rep["gap_source"] = "sample_numbers.npy"
    else:
        gap_after = np.flatnonzero(np.diff(ts) > 1.5 / fs)   # forward gap between sample i and i+1
        rep["gap_source"] = "timestamp steps > 1.5 / fs (no sample numbers given)"
    k = np.clip(np.floor(pos).astype(np.int64), 0, max(ts.size - 2, 0))
    in_gap = np.isin(k, gap_after) & (pos > k) & ~before & ~after
    if return_sides:
        return pos + first_sample, in_gap, before, after
    return pos + first_sample, in_gap, before | after


def timestamps_to_seconds(timestamps, fs, max_backward_s=None, report=None):
    """
    Return Open Ephys timestamps in seconds. Older Open Ephys versions store sample
    counts (step 1) in timestamps.npy; those are divided by fs. Raises on non-finite
    values, repeated timestamps and backward steps of max_backward_s or more
    (default SYNC_CORRECTION_MAX_S). Smaller backward steps are Open Ephys synchronization
    corrections (the stream isn't the main sync stream): accepted and, if `report` is a dict,
    counted there (n_backward_corrections, largest_backward_correction_s).
    """
    import numpy as np
    max_back = SYNC_CORRECTION_MAX_S if max_backward_s is None else float(max_backward_s)
    ts = np.asarray(timestamps, dtype=np.float64)
    if not np.all(np.isfinite(ts)):
        raise ValueError("Timestamps contain NaN or inf; rerun S3 and check timestamp_gap_report.json.")
    counts = ts.size > 1 and abs(float(np.median(np.diff(ts[:1_000_001]))) - 1.0) < 1e-6
    n_corr, largest = 0, 0.0
    if ts.size > 1:
        d = np.diff(ts)
        if counts:
            bad = np.flatnonzero(d <= 0)
            corr = np.zeros(0, dtype=np.int64)
        else:
            bad = np.flatnonzero((d == 0) | (d <= -max_back))
            corr = np.flatnonzero((d < 0) & (d > -max_back))
        if bad.size:
            raise ValueError(
                f"Timestamps are not strictly increasing: they repeat or run backwards by >= {1e3 * max_back:g} ms "
                f"({bad.size} step(s), "
                f"first at sample {int(bad[0])}, step {float(d[bad[0]]):.6g}). See timestamp_gap_report.json."
            )
        n_corr = int(corr.size)
        largest = float(-d[corr].min()) if corr.size else 0.0
        del d
    if report is not None:
        report.update(n_backward_corrections=n_corr, largest_backward_correction_s=largest,
                      max_backward_allowed_s=max_back)
    if counts:
        return ts / float(fs), True
    return ts, False


def _file_sha1(path, block=64 * 1024 * 1024):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


PHY_RECORD_NAME = "nam_phy_curation_record.json"
LABEL_FILES = ("cluster_group.tsv", "cluster_groups.csv", "cluster_KSLabel.tsv", "cluster_info.tsv")


def verify_int16_input(dat_path, si_save_path):
    """
    Check that C3's 16-bit file was written from the CURRENT spike-band folder (S2) with
    its current settings, and has the expected size. Returns the file's settings dict.
    """
    import json
    from pathlib import Path
    dat_path, si_save_path = Path(dat_path), Path(si_save_path)
    sp = dat_path.with_name(dat_path.name + ".settings.json")
    if not dat_path.exists() or not sp.exists():
        raise FileNotFoundError(f"Missing 16-bit file or its settings: {dat_path}\nRun C3.")
    info = json.loads(sp.read_text())
    s2_stamp = si_save_path / SETTINGS_STAMP_NAME
    if Path(info.get("source", "")) != si_save_path:
        raise ValueError(f"The 16-bit file was written from {info.get('source')}, not {si_save_path}. Rerun C3.")
    if not s2_stamp.exists() or info.get("source_settings") != json.loads(s2_stamp.read_text()):
        raise ValueError(
            "The 16-bit file was written from a spike-band folder with different settings than the "
            "current S2 folder. Delete it and rerun C3."
        )
    if dat_path.stat().st_size != int(info["n_channels"]) * int(info["n_frames"]) * 2:
        raise ValueError(f"{dat_path} size doesn't match its settings file. Delete it and rerun C3.")
    return info


def sort_id(sorter_output_folder):
    """Short ID of the current spike times + cluster assignments (used in folder names)."""
    import hashlib, json
    fp = sort_fingerprint(sorter_output_folder)
    return hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:12]


def label_fingerprint(sorter_output_folder):
    """SHA-1 of every cluster-label file present (these decide which units CellExplorer loads)."""
    from pathlib import Path
    out = Path(sorter_output_folder)
    return {name: (_file_sha1(out / name) if (out / name).exists() else None) for name in LABEL_FILES}


def _dat_identity(dat_settings_path):
    """Identity of C3's 16-bit file: its settings file content (source folder + settings)."""
    import json
    from pathlib import Path
    if dat_settings_path is None:
        return None
    return json.loads(Path(dat_settings_path).read_text())


def bombcell_session_name(sorter_output_folder, dat_settings_path):
    """
    Bombcell output folder name. Changes with spike times, cluster assignments and the
    16-bit input file, because Bombcell caches raw waveforms and duplicate-spike results
    in its output folder and would otherwise reuse them for a different sort or input.
    """
    import hashlib, json
    ident = {"sort": sort_fingerprint(sorter_output_folder), "dat": _dat_identity(dat_settings_path)}
    return "bombcell_" + hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def cellexplorer_session_name(sorter_output_folder, dat_settings_path=None):
    """
    CellExplorer session folder name. Changes whenever spike times, cluster assignments,
    the label files or (if given) the 16-bit input file change, so CellExplorer's caches (e.g. *.mono_res.cellinfo.mat)
    are never reused for a different set of units.
    """
    import hashlib, json
    ident = {"sort": sort_fingerprint(sorter_output_folder), "labels": label_fingerprint(sorter_output_folder)}
    if dat_settings_path is not None:
        ident["dat"] = _dat_identity(dat_settings_path)
    return "cellexplorer_" + hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]


def register_phy_curation(kilosort_folder):
    """
    Record a Phy-curated version of the current Kilosort run (after merges / splits).

    Validates that only cluster assignments changed: spike_times.npy must be identical
    to the one Kilosort produced (S4 run record), and spike_clusters.npy must have one
    integer per spike. The S4 run record is left untouched; this writes a separate
    nam_phy_curation_record.json linking the curated version to it.
    """
    import json, datetime
    import numpy as np
    from pathlib import Path
    kilosort_folder = Path(kilosort_folder)
    out = kilosort_folder / "sorter_output"
    rr_path = kilosort_folder / "nam_kilosort_run_record.json"
    if not rr_path.exists():
        raise FileNotFoundError(f"No Kilosort run record: {rr_path}\nRerun S4 with this notebook version.")
    record = json.loads(rr_path.read_text())
    base = record.get("sort_fingerprint")
    if not base:
        raise ValueError("The Kilosort run record has no sort fingerprint. Rerun S4 with this notebook version.")
    current = sort_fingerprint(out)
    if current["spike_times_sha1"] != base["spike_times_sha1"]:
        raise ValueError(
            "spike_times.npy differs from what Kilosort produced. Phy merges and splits never change "
            "spike times, so this is not a Phy curation of this sort. Rerun S4."
        )
    if current == base:
        raise ValueError("spike_clusters.npy is unchanged since Kilosort: there is no Phy curation to register.")
    st = np.load(out / "spike_times.npy", mmap_mode="r")
    clu = np.load(out / "spike_clusters.npy", mmap_mode="r")
    if clu.ravel().size != st.ravel().size:
        raise ValueError(f"spike_clusters.npy has {clu.size} entries for {st.size} spikes.")
    if not np.issubdtype(clu.dtype, np.integer):
        raise ValueError(f"spike_clusters.npy has dtype {clu.dtype}; expected integers.")
    ids = np.unique(clu)
    phy = {
        "registered_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "base_sort_fingerprint": base,
        "curated_sort_fingerprint": current,
        "labels": label_fingerprint(out),
        "n_clusters": int(ids.size),
        "n_clusters_original": int(record.get("n_units", -1)),
    }
    (kilosort_folder / PHY_RECORD_NAME).write_text(json.dumps(phy, indent=2))
    return phy


def verify_sort_input(kilosort_folder, si_save_path):
    """
    Check that the saved spike-band folder (S2) is the exact input of the current
    Kilosort run, using S4's run record:
        input_recording == si_save_path
        input_settings  == the S2 folder's current settings file
        sorter_output   == Kilosort's original output (sort_fingerprint), OR a Phy-curated
                           version of it registered with register_phy_curation()
    Returns the run record (with "phy_curation" added if the curated version is in use).
    Raises with what to rerun otherwise.
    """
    import json
    from pathlib import Path
    kilosort_folder, si_save_path = Path(kilosort_folder), Path(si_save_path)
    rr_path = kilosort_folder / "nam_kilosort_run_record.json"
    stamp_path = si_save_path / SETTINGS_STAMP_NAME
    if not rr_path.exists():
        raise FileNotFoundError(f"No Kilosort run record: {rr_path}\nRerun S4 with this notebook version.")
    if not stamp_path.exists():
        raise FileNotFoundError(f"The spike-band folder has no settings file: {stamp_path}\nRerun S2.")
    record = json.loads(rr_path.read_text())
    if Path(record.get("input_recording", "")) != si_save_path:
        raise ValueError(f"Kilosort sorted {record.get('input_recording')}, not {si_save_path}. Rerun S4.")
    if record.get("input_settings") != json.loads(stamp_path.read_text()):
        raise ValueError(
            "The spike-band folder (S2) has changed since Kilosort sorted it (different settings). "
            "Waveforms and metrics from it would not match the spikes. Rerun S4 on the current S2 "
            "folder, or restore the folder Kilosort sorted."
        )
    current = sort_fingerprint(kilosort_folder / "sorter_output")
    if record.get("sort_fingerprint") == current:
        return record
    phy_path = kilosort_folder / PHY_RECORD_NAME
    if phy_path.exists():
        phy = json.loads(phy_path.read_text())
        if phy.get("base_sort_fingerprint") == record.get("sort_fingerprint") and phy.get("curated_sort_fingerprint") == current:
            return {**record, "phy_curation": phy}
    if record.get("sort_fingerprint", {}).get("spike_times_sha1") == current["spike_times_sha1"]:
        raise ValueError(
            "spike_clusters.npy has changed since Kilosort (Phy merges / splits?) and this version "
            "isn't registered. If you curated in Phy, run cell S5 (Register Phy curation), then C1–C5."
        )
    raise ValueError("sorter_output has changed since the Kilosort run record was written. Rerun S4.")


def sort_fingerprint(sorter_output_folder):
    """
    Identity of one Kilosort run: SHA-1 of spike_times.npy and spike_clusters.npy.
    Every curation output stores this so results from an older sort are never
    merged with a newer one, even if cluster IDs happen to coincide.
    """
    from pathlib import Path
    out = Path(sorter_output_folder)
    return {
        "spike_times_sha1": _file_sha1(out / "spike_times.npy"),
        "spike_clusters_sha1": _file_sha1(out / "spike_clusters.npy"),
    }


# ------------------------------------------------------------
# Readers for curation outputs. Each returns a DataFrame keyed by
# Kilosort cluster_id, or None if that tool has not been run.
# ------------------------------------------------------------

def read_kilosort_units(sorter_output_folder, fs, n_frames=None):
    """
    One row per cluster in spike_clusters.npy (clusters with spikes; stale rows in the
    label files, e.g. clusters merged away in Phy, are ignored):
    cluster_id, ks_label, phy_label, n_spikes, firing_rate_hz.

    ks_label  : Kilosort's automatic label (cluster_KSLabel.tsv, never edited by Phy).
    phy_label : the label in cluster_group.tsv. Kilosort writes it equal to KSLabel;
                Phy overwrites it when you label clusters by hand (good / mua / noise).
                NaN if cluster_group.tsv doesn't exist.
    phy_label_edited : True where phy_label differs from ks_label (a manual decision).

    Firing rate uses n_frames / fs as the duration if given, else last spike time.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    out = Path(sorter_output_folder)
    clu = np.load(out / "spike_clusters.npy").ravel()
    st = np.load(out / "spike_times.npy").ravel()
    ids, counts = np.unique(clu, return_counts=True)
    df = pd.DataFrame({"cluster_id": ids.astype(int), "n_spikes": counts.astype(int)})

    for name, col, new in (("cluster_KSLabel.tsv", "KSLabel", "ks_label"),
                           ("cluster_group.tsv", "group", "phy_label")):
        p = out / name
        if p.exists():
            lab = pd.read_csv(p, sep="\t").rename(columns={col: new})[["cluster_id", new]]
            lab[new] = lab[new].astype("string").str.strip().str.lower()
            df = df.merge(lab, on="cluster_id", how="left")   # only clusters that have spikes
        else:
            df[new] = pd.Series(pd.NA, index=df.index, dtype="string")

    df["phy_label_edited"] = (df["phy_label"].notna() & (df["phy_label"] != df["ks_label"]).fillna(True)).astype(bool)
    df["n_spikes"] = df["n_spikes"].fillna(0).astype(int)
    duration_s = n_frames / fs if n_frames else (float(st.max()) / fs if st.size else np.nan)
    df["firing_rate_hz"] = df["n_spikes"] / duration_s
    return df.sort_values("cluster_id").reset_index(drop=True)


def read_unit_refine(unit_refine_folder):
    """
    UnitRefine labels with real unit IDs, or None if not run.
    Old files that used made-up sequential IDs (column 'Unit_ID') are ignored
    with a warning, because their IDs cannot be trusted.
    """
    import pandas as pd
    from pathlib import Path
    p = Path(unit_refine_folder) / "unit_refine_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "unit_id" not in df.columns:
        print(f"[warning] {p} is from the old UnitRefine code (sequential IDs, not real "
              "cluster IDs). Ignored. Rerun the UnitRefine cell to replace it.")
        return None
    df = df.rename(columns={
        "unit_id": "cluster_id",
        "Noise_Label": "ur_noise_label",
        "Noise_Probability": "ur_noise_probability",
        "SUA_MUA_Label": "ur_sua_mua_label",
        "SUA_MUA_Probability": "ur_sua_mua_probability",
    })
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def read_bombcell(*folders):
    """
    Bombcell unit types (cluster_bc_unitType.tsv) from the first folder that has one,
    or None if Bombcell has not been run.
    """
    import pandas as pd
    from pathlib import Path
    for f in folders:
        if f is None:
            continue
        p = Path(f) / "cluster_bc_unitType.tsv"
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        col = "bc_unitType" if "bc_unitType" in df.columns else df.columns[-1]
        df = df.rename(columns={col: "bc_unit_type"})[["cluster_id", "bc_unit_type"]]
        df["cluster_id"] = df["cluster_id"].astype(int)
        print(f"Bombcell labels read from: {p}")
        return df
    return None


def read_cellexplorer(folder, pattern="*.cell_metrics.cellinfo.mat"):
    """
    CellExplorer putative cell types keyed by Kilosort cluster ID (cluID),
    or None if no cell_metrics file is found. If several match, the newest is used.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy.io import loadmat
    files = sorted(Path(folder).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    path = files[-1]
    cm = loadmat(path, simplify_cells=True)["cell_metrics"]
    if "cluID" in cm:
        ids = np.atleast_1d(cm["cluID"]).astype(int)
    else:
        raise RuntimeError(
            f"{path} has no cluID field, so cell types can't be matched to Kilosort cluster IDs.")
    types = np.atleast_1d(cm["putativeCellType"])
    types = [str(t) for t in types]
    if len(types) != len(ids):
        raise RuntimeError(f"{path}: {len(ids)} cluIDs but {len(types)} cell types.")
    print(f"CellExplorer cell types read from: {path}")
    return pd.DataFrame({"cluster_id": ids, "ce_putative_cell_type": types})


def build_unit_summary(ks_df, ur_df=None, bc_df=None, ce_df=None):
    """
    Merge everything that is available into one table, one row per Kilosort cluster.

    Quality call per tool (only for tools that were run):
        Kilosort   : phy_label == "good" if cluster_group.tsv exists (your manual Phy
                     label wins), else ks_label == "good"
        UnitRefine : ur_noise_label == "neural" and ur_sua_mua_label == "sua"
                     (a missing or unexpected label is never good)
        Bombcell   : bc_unit_type == "GOOD"
    consensus_good = True only if every tool that was run says good AND the unit was
    not manually rejected in Phy (phy_label "noise" or "mua"; manually_rejected column).
    n_quality_sources tells you how many tools contributed.

    Returns (summary_df, sources_used).
    """
    import pandas as pd
    df = ks_df.copy()
    auto = df["ks_label"].astype("string").str.lower()
    manual = df["phy_label"].astype("string").str.lower() if "phy_label" in df else auto
    final = manual.where(manual.notna(), auto)          # manual label wins where it exists
    calls = {"kilosort": final.eq("good").fillna(False).astype(bool)}
    df["manually_rejected"] = (manual.isin(["noise", "mua"]) & (manual != auto).fillna(True)).fillna(False).astype(bool)
    sources = ["kilosort"]

    if ur_df is not None:
        df = df.merge(ur_df, on="cluster_id", how="left")
        noise_lab = df["ur_noise_label"].astype("string").str.strip().str.lower()
        sua_lab = df["ur_sua_mua_label"].astype("string").str.strip().str.lower()
        bad = (~noise_lab.isin(["neural", "noise"]).fillna(False)) | (~sua_lab.isin(["sua", "mua"]).fillna(False))
        if bad.any():
            print(f"[warning] UnitRefine: {int(bad.sum())} unit(s) with a missing or unexpected label "
                  f"(e.g. cluster {df.loc[bad, 'cluster_id'].iloc[0]}); they are not counted as good.")
        calls["unitrefine"] = (noise_lab.eq("neural") & sua_lab.eq("sua")).fillna(False).astype(bool)
        sources.append("unitrefine")
    if bc_df is not None:
        df = df.merge(bc_df, on="cluster_id", how="left")
        calls["bombcell"] = df["bc_unit_type"].astype(str).str.upper().eq("GOOD")
        sources.append("bombcell")
    if ce_df is not None:
        df = df.merge(ce_df, on="cluster_id", how="left")
        sources.append("cellexplorer")

    for name, s in calls.items():
        df[f"good_{name}"] = s.values
    good_cols = [f"good_{n}" for n in calls]
    df["n_quality_sources"] = len(good_cols)
    df["n_sources_good"] = df[good_cols].sum(axis=1).astype(int)
    df["consensus_good"] = df[good_cols].all(axis=1) & ~df["manually_rejected"]
    return df, sources


def _whole_nonnegative_int64(path, what):
    """
    Load a 1-D .npy of indices/IDs as int64, refusing anything an integer cast would
    silently change: NaN/inf, fractional values, negative values, non-numeric dtypes or
    values beyond the int64 range. Integer files and float files holding whole numbers
    are accepted. (Added 2026-10-29.)
    """
    import numpy as np
    a = np.asarray(np.load(path, allow_pickle=False)).ravel()
    if a.dtype.kind == "b" or a.dtype.kind not in "iuf":
        raise TypeError(f"{path}: {what} have dtype {a.dtype}; expected integers.")
    if a.dtype.kind == "f":
        bad = ~np.isfinite(a)
        if bad.any():
            raise ValueError(f"{path}: {int(bad.sum())} {what} are NaN or inf "
                             f"(first at position {int(np.flatnonzero(bad)[0])}).")
        frac = a != np.round(a)
        if frac.any():
            i = int(np.flatnonzero(frac)[0])
            raise ValueError(f"{path}: {int(frac.sum())} {what} are not whole numbers "
                             f"(e.g. {float(a[i])!r} at position {i}).")
        if a.size and a.max() >= 2.0 ** 63:
            raise ValueError(f"{path}: {what} exceed the int64 range.")
    elif a.dtype.kind == "u" and a.size and a.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{path}: {what} exceed the int64 range.")
    if a.size and a.min() < 0:
        raise ValueError(f"{path}: {int((a < 0).sum())} negative {what} (min {a.min()}).")
    return a.astype(np.int64)


def load_spike_samples(path, n_frames=None):
    """
    Kilosort spike_times.npy as validated int64 sample indices (see _whole_nonnegative_int64).
    With n_frames, also refuses indices at or past the end of the recording.
    """
    out = _whole_nonnegative_int64(path, "spike times")
    if n_frames is not None and out.size and out.max() >= int(n_frames):
        raise ValueError(
            f"{path}: spike times reach sample {int(out.max()):,}, past the end of the "
            f"{int(n_frames):,}-frame recording. The Kilosort output and the saved recording "
            "come from different runs or crops."
        )
    return out


def load_spike_clusters(path, n_spikes=None):
    """Kilosort spike_clusters.npy as validated int64 cluster IDs; optional length check."""
    out = _whole_nonnegative_int64(path, "cluster IDs")
    if n_spikes is not None and out.size != int(n_spikes):
        raise ValueError(f"{path} has {out.size:,} entries but spike_times.npy has {int(n_spikes):,}.")
    return out


# ============================================================
# ANALOG (ADC) STREAM: discovery, loading, shared-clock check, crop (added 2026-10-30)
# Any Open Ephys continuous stream other than the probe, e.g. a OneBox ADC stream
# ('OneBox-121.OneBox-ADC') or an NI-DAQ stream ('NI-DAQmx-104.PCIe-6251').
# Nothing here writes, repairs or replaces acquisition files: missing files stop with
# their exact path, and timing is only taken from the stream's own timestamps.npy.
# ============================================================

ADC_REQUIRED_FILES = ("continuous.dat", "timestamps.npy", "sample_numbers.npy")
_VOLTAGE_UNITS_TO_UV = {"V": 1e6, "mV": 1e3, "uV": 1.0, "µV": 1.0}


def parse_stream_folder(folder_name):
    """'OneBox-121.OneBox-ADC' -> {'processor': 'OneBox', 'processor_id': 121, 'stream': 'OneBox-ADC'} (None if no match)."""
    import re
    m = re.match(r"^(.*)-(\d+)\.(.+)$", str(folder_name).rstrip("/"))
    if not m:
        return None
    return {"processor": m.group(1), "processor_id": int(m.group(2)), "stream": m.group(3)}


def list_openephys_streams(recording_root, experiment_label):
    """
    Signal streams the installed SpikeInterface/Neo reader finds in ONE recording folder
    (recording_root = <session>/<Record Node>/<experiment>/<recording>).
    Returns (stream_names, stream_ids). Names look like '<Record Node>#<folder>'; Neo puts
    channels named 'ADC...' in a separate stream '<Record Node>#<folder>_ADC'.
    """
    from pathlib import Path
    from spikeinterface.extractors import get_neo_streams
    recording_root = Path(recording_root)
    if not (recording_root / "structure.oebin").is_file():
        raise FileNotFoundError(
            f"structure.oebin not found: {recording_root / 'structure.oebin'}\n"
            "It is written by the Open Ephys GUI during recording; check base_path, record_node, "
            "experiment_label and recording_label in 0.2."
        )
    # The reader opens EVERY continuous stream listed in structure.oebin; a missing file in any of
    # them (also an ADC stream not selected in ADC_MODE) stops it with a bare file error. Say so.
    import json as _json
    try:
        _entries = _json.loads((recording_root / "structure.oebin").read_text(encoding="utf-8")).get("continuous", [])
    except ValueError as exc:
        raise ValueError(f"structure.oebin could not be read: {recording_root / 'structure.oebin'} ({exc})") from None
    _missing = []
    for _e in _entries:
        _d = recording_root / "continuous" / str(_e.get("folder_name", "")).rstrip("/")
        if not (_d / "continuous.dat").is_file():
            _missing.append(str(_d / "continuous.dat"))
        if not (_d / "sample_numbers.npy").is_file() and not (_d / "timestamps.npy").is_file():
            _missing.append(str(_d / "sample_numbers.npy") + " (and timestamps.npy)")
    if _missing:
        raise FileNotFoundError(
            "The Open Ephys reader opens every continuous stream listed in structure.oebin, and these "
            "files are missing:\n  " + "\n  ".join(_missing)
            + "\nA missing file in any stream folder (also one not selected in ADC_MODE) stops reading this "
              "recording. Restore the files from the original recording; this notebook does not reconstruct them."
        )
    names, ids = get_neo_streams("openephysbinary", folder_path=str(recording_root),
                                 experiment_names=experiment_label)
    return [str(n) for n in names], [str(i) for i in ids]


def resolve_adc_stream(recording_root, experiment_label, record_node, adc_label, stream_name=None,
                       probe_label=None):
    """
    Map the configured ADC folder name to the reader's stream identifier.

    Candidates are '<record_node>#<adc_label>' and '<record_node>#<adc_label>_ADC' (Neo's name
    when the channels are called ADC0, ADC1, ... as in a OneBox). If stream_name is given it
    must be one of those and present. Raises with the expected path and the available names if
    the folder or stream is missing, if the folder holds two reader streams, or if the reader
    found streams of another Record Node (wrong folder selected).
    Returns (stream_name, stream_id, {name: id for all streams}).
    """
    from pathlib import Path
    recording_root = Path(recording_root)
    if not adc_label:
        raise ValueError("ADC folder label is not set (ADC_SOURCES[...]['label'] in cell 0.2).")
    if probe_label is not None and adc_label == probe_label:
        raise ValueError(f"ADC label {adc_label!r} is the probe stream. Set ADC_SOURCES[...]['label'] to the analog stream's folder.")
    cont = recording_root / "continuous"
    folder = cont / adc_label
    if not folder.is_dir():
        available = sorted(p.name for p in cont.iterdir() if p.is_dir()) if cont.is_dir() else []
        raise FileNotFoundError(
            f"ADC stream folder not found:\n{folder}\n"
            f"Stream folders in {cont}: {available}\n"
            "Set ADC_SOURCES[...]['label'] in 0.2 to one of these, or choose another ADC_MODE ('off' = none)."
        )
    missing = [str(folder / f) for f in ADC_REQUIRED_FILES if not (folder / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required ADC acquisition files are missing:\n  " + "\n  ".join(missing)
            + "\nThey are written by the Open Ephys GUI (>= 0.6) during recording. This notebook does not "
              "reconstruct them. Restore them from the original recording, or choose another ADC_MODE in 0.2."
        )
    names, ids = list_openephys_streams(recording_root, experiment_label)
    foreign = [n for n in names if "#" in n and not n.startswith(f"{record_node}#")]
    if foreign:
        raise ValueError(f"The reader found streams of another Record Node in {recording_root}: {foreign}. "
                         "Check record_node in 0.2.")
    base = f"{record_node}#{adc_label}"
    allowed = (base, base + "_ADC")
    if stream_name is not None:
        if stream_name not in names or stream_name not in allowed:
            raise ValueError(
                f"Stream {stream_name!r} is not a stream of folder {adc_label!r} in this recording.\n"
                f"Streams found: {names}"
            )
        chosen = stream_name
    else:
        cands = [n for n in names if n in allowed]
        if not cands:
            raise ValueError(
                f"The reader finds no stream for folder {adc_label!r} (expected {base!r} or {base + '_ADC'!r}).\n"
                f"Streams found: {names}\nIs the folder listed in structure.oebin?"
            )
        if len(cands) > 1:
            raise ValueError(
                f"Folder {adc_label!r} holds two reader streams: {cands} (ADC-named and other channels). "
                "Set ADC_SOURCES[...]['stream_name'] in 0.2 to the one to load."
            )
        chosen = cands[0]
    return chosen, ids[names.index(chosen)], dict(zip(names, ids))


def read_oebin_continuous(recording_root, folder_name):
    """The structure.oebin entry of one continuous stream folder (raises if not listed)."""
    import json
    from pathlib import Path
    oebin = Path(recording_root) / "structure.oebin"
    if not oebin.is_file():
        raise FileNotFoundError(f"structure.oebin not found: {oebin}")
    d = json.loads(oebin.read_text(encoding="utf-8"))
    for entry in d.get("continuous", []):
        if entry.get("folder_name", "").rstrip("/") == folder_name:
            return entry
    listed = [e.get("folder_name", "").rstrip("/") for e in d.get("continuous", [])]
    raise ValueError(f"{folder_name!r} is not listed in {oebin}. Continuous streams listed: {listed}")


def _oebin_ttl_dirs(recording_root, folder_name):
    """Event folders of one stream that hold TTL states (timestamps.npy + states.npy)."""
    import json
    from pathlib import Path
    recording_root = Path(recording_root)
    out = []
    try:
        d = json.loads((recording_root / "structure.oebin").read_text(encoding="utf-8"))
    except Exception:
        return out
    for e in d.get("events", []):
        fn = e.get("folder_name", "")
        if fn.startswith(folder_name + "/"):
            p = recording_root / "events" / fn.rstrip("/")
            if (p / "timestamps.npy").is_file() and (p / "states.npy").is_file():
                out.append(p)
    return out


def scan_stream_clock(timestamps, sample_numbers=None, start=0, end=None, chunk=10_000_000,
                      max_backward_s=0.0, fs=None, jump_tolerance_s=None, max_report=20):
    """
    Check timestamps (and sample numbers) of one stream over [start, end), in chunks (memmaps OK).
    Counts: non-finite, -1 (Open Ephys writes -1 for samples it could not synchronize), other
    negative, and non-increasing timestamps; with sample_numbers also forward gaps (missing
    samples), repeats and backward steps. ok_clock / ok_samples summarize.

    max_backward_s: timestamp steps in (-max_backward_s, 0) are counted separately as small
    corrections and allowed. Open Ephys's synchronized timestamps of a non-main stream jump
    whenever the synchronizer updates its mapping (e.g. 9 steps of -4 to -11 us in a 2-min
    NI-DAQ stream at 40 kHz). align_adc_to_probe() allows SYNC_CORRECTION_MAX_S; larger
    backward steps stop it.

    fs (with sample_numbers): also compare every timestamp step with the sample-number step,
    as check_timestamp_gaps() does for S3. A step is unexplained when
        sample-number step == 1 : |dt - 1/fs| >= jump_tolerance_s (and > half a sample)
        sample-number step  > 1 : |dt - step/fs| > half a sample + jump_tolerance_s
    in either direction (default jump_tolerance_s = SYNC_CORRECTION_MAX_S). Real dropped samples
    (timestamps jump together with the sample numbers) and small Open Ephys corrections pass; a
    jump with continuous sample numbers does not. Reported as n_unexplained_timestamp_jumps
    (+ the first max_report: index, step_s, expected_step_s, sample_number_step); ok_timing.
    Steps touching a non-finite / -1 value or a repeated / backward sample number are counted by
    the other checks, not here.
    """
    import numpy as np
    n = len(timestamps)
    end = n if end is None else int(end)
    start = int(start)
    r = dict(start=start, end=end, n_nonfinite=0, n_unsynchronized=0, n_negative=0, n_nonincreasing=0,
             n_small_backward_corrections=0, largest_small_correction_s=0.0, max_backward_allowed_s=float(max_backward_s),
             first_bad_timestamp_index=None, n_forward_gaps=0, missing_samples_total=0,
             n_repeated_samples=0, n_backward_jumps=0, first_sample_gap_index=None)
    check_jumps = sample_numbers is not None and fs is not None
    if check_jumps:
        fs = float(fs)
        if not (np.isfinite(fs) and fs > 0):
            raise ValueError(f"scan_stream_clock: fs must be a positive number, got {fs!r}")
        tol_j = SYNC_CORRECTION_MAX_S if jump_tolerance_s is None else float(jump_tolerance_s)
        r.update(n_unexplained_timestamp_jumps=0, first_unexplained_timestamp_jumps=[],
                 largest_unexplained_jump_s=0.0, jump_tolerance_s=tol_j, fs_hz=fs)
    prev_t = prev_s = None
    pos = start
    while pos < end:
        stop = min(end, pos + chunk)
        t = np.asarray(timestamps[pos:stop], dtype=np.float64)
        bad = ~np.isfinite(t)
        uns = t == -1.0
        neg = (t < 0) & ~uns
        tt = t if prev_t is None else np.concatenate([[prev_t], t])
        with np.errstate(invalid="ignore"):
            dt = np.diff(tt)
            small = (dt < 0) & (dt > -float(max_backward_s))      # equal timestamps are never a correction
            noninc = ~(dt > 0) & ~small
        if small.any():
            r["n_small_backward_corrections"] += int(small.sum())
            r["largest_small_correction_s"] = max(r["largest_small_correction_s"], float(-dt[small].min()))
        r["n_nonfinite"] += int(bad.sum())
        r["n_unsynchronized"] += int(uns.sum())
        r["n_negative"] += int(neg.sum())
        r["n_nonincreasing"] += int(noninc.sum())
        if r["first_bad_timestamp_index"] is None:
            cand = [pos + int(np.flatnonzero(m)[0]) for m in (bad, uns, neg) if m.any()]
            if noninc.any():
                cand.append(pos + int(np.flatnonzero(noninc)[0]) - (0 if prev_t is None else 1) + 1)
            if cand:
                r["first_bad_timestamp_index"] = min(cand)
        prev_t = float(t[-1])
        if sample_numbers is not None:
            s = np.asarray(sample_numbers[pos:stop]).astype(np.int64)
            ss = s if prev_s is None else np.concatenate([[prev_s], s])
            d = np.diff(ss)
            fwd = d > 1
            r["n_forward_gaps"] += int(fwd.sum())
            r["missing_samples_total"] += int((d[fwd] - 1).sum())
            r["n_repeated_samples"] += int((d == 0).sum())
            r["n_backward_jumps"] += int((d < 0).sum())
            if r["first_sample_gap_index"] is None and (d != 1).any():
                r["first_sample_gap_index"] = pos + int(np.flatnonzero(d != 1)[0]) + (0 if prev_s is None else -1) + 1
            if check_jumps and d.size:
                # dt and d both index step k -> k+1 (with the previous chunk's last sample first)
                expected = d.astype(np.float64) / fs
                with np.errstate(invalid="ignore"):
                    dev = np.abs(dt - expected)
                    ok_t = np.isfinite(tt[:-1]) & np.isfinite(tt[1:]) & (tt[:-1] != -1.0) & (tt[1:] != -1.0)
                    unexp = ok_t & (((d == 1) & (dev >= tol_j) & (dev > 0.5 / fs))
                                    | ((d > 1) & (dev > 0.5 / fs + tol_j)))
                k = np.flatnonzero(unexp)
                if k.size:
                    r["n_unexplained_timestamp_jumps"] += int(k.size)
                    r["largest_unexplained_jump_s"] = max(r["largest_unexplained_jump_s"], float(dev[k].max()))
                    base = pos - (0 if prev_s is None else 1)       # index of the sample before step 0
                    room = max_report - len(r["first_unexplained_timestamp_jumps"])
                    for i in k[:max(0, room)]:
                        r["first_unexplained_timestamp_jumps"].append(
                            {"index": int(base + i), "step_s": float(dt[i]),
                             "expected_step_s": float(expected[i]), "sample_number_step": int(d[i])})
            prev_s = int(s[-1])
        pos = stop
    r["ok_clock"] = (r["n_nonfinite"] == 0 and r["n_unsynchronized"] == 0 and r["n_negative"] == 0
                     and r["n_nonincreasing"] == 0)
    r["ok_samples"] = r["n_repeated_samples"] == 0 and r["n_backward_jumps"] == 0
    if check_jumps:
        r["ok_timing"] = r["n_unexplained_timestamp_jumps"] == 0
    return r


def load_adc_recording(config, channel_ids=None, unit_if_not_stated=None):
    """
    Load the configured analog (ADC) stream in full, with its own sampling rate and gains.

    config            : nam.load_experiment_config(config_path) (needs ADC_label; load_adc True)
    channel_ids       : IDs to keep, as the reader names them (e.g. ['ADC0', 'ADC3'] or ['AI0']);
                        None = all. Unknown IDs raise and list the IDs found.
    unit_if_not_stated: 'V', 'mV' or 'uV' to use for channels whose structure.oebin 'units' is
                        empty, ONLY if you know the unit (e.g. from the device documentation).
                        None (default): such channels keep raw values only (gain_to_uV = NaN,
                        physical_unit 'not stated').
    The Neuropixels 0.195 uV/bit value is never used here. Nothing is filtered, referenced,
    phase-shifted or sorted. Returns (recording, info dict).
    """
    import numpy as np
    from pathlib import Path
    from spikeinterface.extractors import read_openephys

    if not config.get("load_adc"):
        raise ValueError("ADC loading is off in the configuration (ADC_MODE = 'off' in 0.2).")
    label = config.get("ADC_label")
    recording_root = (Path(config["base_path"]) / config["record_node"] / config["experiment_label"]
                      / config["recording_label"])
    stream_name, stream_id, all_streams = resolve_adc_stream(
        recording_root, config["experiment_label"], config["record_node"], label,
        stream_name=config.get("adc_stream_name"), probe_label=config.get("Neuropix_PXI_label"))
    if config.get("adc_stream_id") is not None and str(config["adc_stream_id"]) != str(stream_id):
        raise ValueError(
            f"Stream {stream_name!r} now has reader ID {stream_id}, but 0.2 resolved it as "
            f"{config['adc_stream_id']}. The stream list changed: rerun 0.2."
        )
    folder = recording_root / "continuous" / label
    files = {name: folder / name for name in ADC_REQUIRED_FILES}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required ADC acquisition files are missing:\n  " + "\n  ".join(missing)
            + "\nThey are written by the Open Ephys GUI (>= 0.6) during recording. This notebook "
              "does not reconstruct them. Restore them from the original recording, or choose another ADC_MODE in 0.2."
        )
    entry = read_oebin_continuous(recording_root, label)

    # all_annotations stays False: Neo 0.14 mis-slices per-channel annotations of ADC-only streams
    # (e.g. OneBox ADC0-ADC11) and the reader then fails
    rec = read_openephys(folder_path=str(recording_root), experiment_names=config["experiment_label"],
                         stream_name=stream_name, load_sync_timestamps=False, all_annotations=False)
    if rec.get_num_segments() != 1:
        raise ValueError(f"{stream_name}: {rec.get_num_segments()} segments; expected 1 (one recording folder).")
    n = int(rec.get_num_samples())
    fs = float(rec.get_sampling_frequency())
    ts = np.load(files["timestamps.npy"], mmap_mode="r")
    sn = np.load(files["sample_numbers.npy"], mmap_mode="r")
    problems = []
    if ts.ndim != 1 or ts.dtype.kind != "f":
        problems.append(f"timestamps.npy: shape {ts.shape}, dtype {ts.dtype} (expected 1-D float seconds)")
    if sn.ndim != 1 or sn.dtype.kind not in "iu":
        problems.append(f"sample_numbers.npy: shape {sn.shape}, dtype {sn.dtype} (expected 1-D integers)")
    if len(ts) != n or len(sn) != n:
        problems.append(f"lengths: continuous.dat {n:,} samples, timestamps.npy {len(ts):,}, sample_numbers.npy {len(sn):,}")
    n_ch_file = int(entry["num_channels"])
    expected_bytes = n * n_ch_file * 2
    if files["continuous.dat"].stat().st_size != expected_bytes:
        problems.append(f"continuous.dat is {files['continuous.dat'].stat().st_size:,} bytes; "
                        f"{n:,} samples x {n_ch_file} channels x 2 bytes = {expected_bytes:,}")
    if abs(float(entry["sample_rate"]) - fs) > 1e-6:
        problems.append(f"sample rate: reader {fs} Hz, structure.oebin {entry['sample_rate']} Hz")
    if problems:
        raise ValueError(f"ADC stream {folder} is inconsistent:\n  " + "\n  ".join(problems))

    found_ids = [str(c) for c in rec.channel_ids]
    if channel_ids is not None:
        channel_ids = [str(c) for c in channel_ids]
        unknown = [c for c in channel_ids if c not in found_ids]
        if unknown or not channel_ids:
            raise ValueError(f"ADC channel_ids {unknown or channel_ids} not in this stream. Channel IDs found: {found_ids}")
        rec = rec.select_channels(channel_ids)

    if unit_if_not_stated is not None and unit_if_not_stated not in _VOLTAGE_UNITS_TO_UV:
        raise ValueError(f"unit_if_not_stated must be one of {list(_VOLTAGE_UNITS_TO_UV)} or None.")
    by_name = {c["channel_name"]: c for c in entry["channels"]}
    units, gains, offsets, phys, gphys, status_list = [], [], [], [], [], []
    for ch in rec.channel_ids:
        info = by_name.get(str(ch))
        if info is None:
            raise ValueError(f"Channel {ch} is not in structure.oebin for {label}.")
        stated = (info.get("units") or "").strip()
        bv = float(info["bit_volts"])
        if stated in _VOLTAGE_UNITS_TO_UV:
            unit, g, status = stated, bv * _VOLTAGE_UNITS_TO_UV[stated], f"stated in structure.oebin ({stated})"
        elif stated:
            unit, g, status = stated, float("nan"), f"stated in structure.oebin ({stated}, not a voltage unit)"
        elif unit_if_not_stated is not None:
            unit, g = unit_if_not_stated, bv * _VOLTAGE_UNITS_TO_UV[unit_if_not_stated]
            status = f"not stated in structure.oebin; {unit_if_not_stated} set by unit_if_not_stated (ADC_SOURCES in 0.2)"
        else:
            unit, g, status = "not stated", float("nan"), "not stated in structure.oebin: raw values only"
        units.append({"channel_id": str(ch), "oebin_units": stated, "bit_volts": bv,
                      "physical_unit": unit, "gain_to_uV": g, "status": status})
        gains.append(g)
        offsets.append(0.0 if np.isfinite(g) else float("nan"))
        phys.append(unit)
        gphys.append(bv if unit != "not stated" and np.isfinite(g) else float("nan"))
        status_list.append(status)
    rec.set_channel_gains(np.asarray(gains, dtype=float))
    rec.set_channel_offsets(np.asarray(offsets, dtype=float))
    rec.set_property("physical_unit", np.asarray(phys))
    rec.set_property("gain_to_physical_unit", np.asarray(gphys, dtype=float))
    rec.set_property("adc_unit_status", np.asarray(status_list))

    info = {
        "adc_label": label,
        "stream_name": stream_name,
        "stream_id": str(stream_id),
        "folder": str(folder),
        "recording_root": str(recording_root),
        "files": {k: str(v) for k, v in files.items()},
        "sampling_rate_hz": fs,
        "n_samples": n,
        "duration_s": n / fs,
        "channel_ids_found": found_ids,
        "channel_ids_used": [str(c) for c in rec.channel_ids],
        "units": units,
        "reader_has_probe": bool(rec.has_probe()),
        "all_streams": all_streams,
    }
    return rec, info


# Shared-clock evidence inside the retained interval (compare_sync_events with a window):
# fraction of the in-window TTL pulses of EACH stream that must have a partner in the other
# stream within the tolerance, and the longest run of consecutive unmatched in-window pulses
# allowed (an occasional missed event is tolerated; a stretch of disagreeing pulses is not).
SYNC_MIN_FRACTION_IN_WINDOW = 0.9
SYNC_MAX_UNMATCHED_RUN = 3
# Matches must span the retained interval: the first matched pulse no later than this after t0,
# the last no earlier than this before t1, and no longer than this between consecutive matched
# pulses (s). 10 s suits a 1 Hz sync line (allows a few missed pulses); set it from the sync
# line's period (cell 1.2: ADC_SYNC_MAX_GAP_S). Nothing is interpolated across a longer gap:
# the interval is then 'not verified' (insufficient full-interval evidence).
SYNC_MAX_GAP_S = 10.0


def _match_one_to_one(a, p, tol):
    """
    One-to-one, order-preserving matching of two sorted event-time arrays within tol.
    Candidate pairs are each a-event with its nearest p-event on either side (<= 2 per a-event,
    so memory is linear); accepted closest first, each event used at most once; pairs that would
    cross an accepted pair (break time order) are dropped. Duplicate / bouncing events therefore
    count once: three ADC edges within tol of one probe edge give ONE pair. Closest-first can
    pair fewer events than possible when several edges of BOTH streams fall within tol of each
    other; that undercounts the evidence (fails closed) and never counts an event twice.
    Returns (a_matched bool[len(a)], p_matched bool[len(p)], a_idx, p_idx, |diff|) for the pairs,
    sorted by time.
    """
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    am, pm = np.zeros(a.size, bool), np.zeros(p.size, bool)
    empty = (am, pm, np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    if a.size == 0 or p.size == 0:
        return empty
    j = np.searchsorted(p, a)
    ci = np.concatenate([np.arange(a.size), np.arange(a.size)])
    cj = np.concatenate([j - 1, j])
    ok = (cj >= 0) & (cj < p.size)
    ci, cj = ci[ok], cj[ok]
    d = np.abs(a[ci] - p[cj])
    keep = d <= tol
    ci, cj, d = ci[keep], cj[keep], d[keep]
    if ci.size == 0:
        return empty
    order = np.lexsort((ci, d))                    # closest first; ties by time
    acc_a, acc_p = [], []
    for k in order:
        i, jj = int(ci[k]), int(cj[k])
        if am[i] or pm[jj]:
            continue
        am[i] = pm[jj] = True
        acc_a.append(i); acc_p.append(jj)
    ia, ip = np.asarray(acc_a, int), np.asarray(acc_p, int)
    o = np.argsort(ia)
    ia, ip = ia[o], ip[o]
    # keep the pairs whose probe indices stay increasing (drop crossings; tol << pulse spacing
    # normally, so this only acts on clustered duplicates)
    good = np.ones(ia.size, bool)
    last = -1
    for k in range(ia.size):
        if ip[k] > last:
            last = ip[k]
        else:
            good[k] = False
            am[ia[k]] = pm[ip[k]] = False
    ia, ip = ia[good], ip[good]
    return am, pm, ia, ip, np.abs(a[ia] - p[ip])


def _nearest_abs_diff(x, ref):
    """|x - nearest value of sorted ref| for every x (inf if ref is empty)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if ref.size == 0:
        return np.full(x.size, np.inf)
    if ref.size == 1:
        return np.abs(x - ref[0])
    idx = np.clip(np.searchsorted(ref, x), 1, ref.size - 1)
    return np.minimum(np.abs(x - ref[idx - 1]), np.abs(x - ref[idx]))


def _longest_false_run(m):
    """Length of the longest run of False in a boolean array."""
    import numpy as np
    m = np.asarray(m, dtype=bool)
    if m.size == 0 or m.all():
        return 0
    padded = np.concatenate([[True], m, [True]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))       # starts (-1) and ends (+1) of False runs
    return int((edges[1::2] - edges[0::2]).max())


def compare_sync_events(recording_root, probe_label, adc_label, tolerance_s, window=None, min_matches=3,
                        min_fraction=None, max_unmatched_run=None, max_gap_s=None):
    """
    Evidence that the probe and ADC timestamps share one clock: the same physical TTL pulses
    (e.g. the sync line recorded by both devices) must carry the same timestamp in both streams.
    Every rising-edge line of the probe's TTL events is compared with every line of the ADC's.
    Pulses are paired one-to-one in time order (_match_one_to_one): each pulse counts at most
    once, so duplicate or bouncing edges are not counted as independent evidence.

    window = (t0, t1), the retained interval (used by align_adc_to_probe): the decision uses ONLY
    the pulses inside it, so agreement elsewhere in the session cannot hide a local failure.
    A pair is verified when, inside the window,
      - >= min_matches ADC pulses have a probe pulse within tolerance_s,
      - >= min_fraction of the ADC pulses AND >= min_fraction of the probe pulses have a partner
        (ADC -> probe and probe -> ADC; default SYNC_MIN_FRACTION_IN_WINDOW),
      - no more than max_unmatched_run consecutive pulses of either stream are unmatched
        (default SYNC_MAX_UNMATCHED_RUN);
      - the matched pulses span the interval: first match <= max_gap_s after t0, last match
        >= max_gap_s before t1, and <= max_gap_s between consecutive matches (default
        SYNC_MAX_GAP_S). A few pulses at one end of a long crop are therefore NOT verified
        ("insufficient full-interval evidence"); nothing is interpolated.
    Fewer than min_matches in-window pulses -> not verified ("insufficient evidence"), so sparse
    TTL sessions are refused, not accepted. Among verified pairs the most in-window matches win.
    Between matched pulses the timing rests on the streams' own timestamps (checked separately
    by scan_stream_clock). Whole-session counts are reported too.
    Without a window: the older whole-session test (>= min_matches ADC pulses matched and
    >= 90 % of those inside the probe pulses' time span).
    Returns a dict with status 'verified' or 'not_verified' and the details.
    """
    import numpy as np
    min_fraction = SYNC_MIN_FRACTION_IN_WINDOW if min_fraction is None else float(min_fraction)
    max_unmatched_run = SYNC_MAX_UNMATCHED_RUN if max_unmatched_run is None else int(max_unmatched_run)
    max_gap_s = SYNC_MAX_GAP_S if max_gap_s is None else float(max_gap_s)
    if not (np.isfinite(max_gap_s) and max_gap_s > 0):
        raise ValueError(f"max_gap_s must be a positive number of seconds, got {max_gap_s!r}")

    def rising(label):
        out = {}
        for d in _oebin_ttl_dirs(recording_root, label):
            t = np.asarray(np.load(d / "timestamps.npy"), dtype=np.float64)
            s = np.asarray(np.load(d / "states.npy"))
            ok = np.isfinite(t) & (t >= 0)
            for line in np.unique(s[(s > 0) & ok]):
                out[(d.name if d.name != "TTL" else d.parent.name + "/TTL", int(line))] = np.sort(t[(s == line) & ok])
        return out

    p_ev, a_ev = rising(probe_label), rising(adc_label)
    res = {"status": "not_verified", "tolerance_ms": 1000 * tolerance_s,
           "probe_ttl_lines": {f"{k[0]} line {k[1]}": int(v.size) for k, v in p_ev.items()},
           "adc_ttl_lines": {f"{k[0]} line {k[1]}": int(v.size) for k, v in a_ev.items()}}
    if window is not None:
        w0, w1 = float(window[0]), float(window[1])
        res.update(window_s=[w0, w1], min_matches=int(min_matches), min_fraction_in_window=min_fraction,
                   max_unmatched_run=max_unmatched_run, max_gap_s=max_gap_s, matching="one-to-one")
    if not p_ev or not a_ev:
        res["reason"] = ("no TTL events recorded for the " + ("probe" if not p_ev else "ADC") + " stream, so no "
                         "shared pulses can be compared")
        return res
    best, best_fail, closest = None, None, None
    for pk, pt in p_ev.items():
        for ak, at in a_ev.items():
            if pt.size == 0 or at.size == 0:
                continue
            p_name, a_name = f"{pk[0]} line {pk[1]}", f"{ak[0]} line {ak[1]}"
            med = float(np.median(_nearest_abs_diff(at, pt)))
            if closest is None or med < closest[2]:
                closest = (p_name, a_name, med)
            m, mp_all, ia, ip, dpair = _match_one_to_one(at, pt, tolerance_s)
            d = np.full(at.size, np.inf)
            d[ia] = dpair                                       # |difference| of each paired ADC pulse
            n_m = int(m.sum())
            inside = (at >= pt[0] - tolerance_s) & (at <= pt[-1] + tolerance_s)
            cand = {"probe_line": p_name, "adc_line": a_name, "n_matched": n_m,
                    "fraction_matched": n_m / max(1, int(inside.sum()))}
            if window is None:
                if not (n_m >= min_matches and cand["fraction_matched"] >= 0.9):
                    continue
                cand.update(n_matched_in_window=None,
                            max_abs_difference_ms=float(1000 * d[m].max()),
                            median_abs_difference_ms=float(1000 * np.median(d[m])),
                            first_matched_s=float(at[m][0]), last_matched_s=float(at[m][-1]))
                if best is None or n_m > best["n_matched"]:
                    best = cand
                continue
            # ---- decision on the retained interval only, both directions ----
            a_in = (at >= w0) & (at <= w1)
            p_in = (pt >= w0) & (pt <= w1)
            ma = m[a_in]                                            # ADC -> probe, in window
            mp = mp_all[p_in]                                       # probe -> ADC, in window (same pairs)
            n_a, n_p = int(a_in.sum()), int(p_in.sum())
            n_win = int(ma.sum())
            fa, fp = (n_win / n_a if n_a else 0.0), (int(mp.sum()) / n_p if n_p else 0.0)
            run = max(_longest_false_run(ma), _longest_false_run(mp))
            tm = at[a_in][ma]
            dm = d[a_in][ma]
            cand.update(n_matched_in_window=n_win, n_adc_pulses_in_window=n_a, n_probe_pulses_in_window=n_p,
                        n_probe_matched_in_window=int(mp.sum()),
                        fraction_adc_matched_in_window=fa, fraction_probe_matched_in_window=fp,
                        longest_unmatched_run=run,
                        max_abs_difference_ms=float(1000 * dm.max()) if n_win else None,
                        median_abs_difference_ms=float(1000 * np.median(dm)) if n_win else None,
                        first_matched_s=float(tm[0]) if n_win else None,
                        last_matched_s=float(tm[-1]) if n_win else None,
                        first_match_after_t0_s=float(tm[0] - w0) if n_win else None,
                        last_match_before_t1_s=float(w1 - tm[-1]) if n_win else None,
                        largest_interval_between_matches_s=float(np.diff(tm).max()) if n_win > 1 else None)
            problems = []
            if n_win < min_matches:
                problems.append(f"only {n_win} matched pulse(s) inside the retained interval (need {min_matches}; "
                                "insufficient evidence)")
            if fa < min_fraction:
                problems.append(f"{n_win}/{n_a} ADC pulses inside the interval match a probe pulse "
                                f"({100 * fa:.1f} % < {100 * min_fraction:g} %)")
            if fp < min_fraction:
                problems.append(f"{int(mp.sum())}/{n_p} probe pulses inside the interval match an ADC pulse "
                                f"({100 * fp:.1f} % < {100 * min_fraction:g} %)")
            if run > max_unmatched_run:
                problems.append(f"{run} consecutive pulses inside the interval are unmatched "
                                f"(allowed {max_unmatched_run})")
            if n_win:
                gaps = {"after t0 before the first match": tm[0] - w0, "after the last match before t1": w1 - tm[-1],
                        "between consecutive matches (largest)": float(np.diff(tm).max()) if n_win > 1 else 0.0}
                long_ = [f"{v:.1f} s {k}" for k, v in gaps.items() if v > max_gap_s]
                if long_:
                    problems.append("insufficient full-interval evidence: " + ", ".join(long_)
                                    + f" (allowed {max_gap_s:g} s; ADC_SYNC_MAX_GAP_S)")
            cand["problems"] = problems
            if not problems:
                if best is None or (n_win, n_m) > (best["n_matched_in_window"], best["n_matched"]):
                    best = cand
            elif n_m >= 1 and (best_fail is None or (n_win, n_m) > (best_fail["n_matched_in_window"],
                                                                              best_fail["n_matched"])):
                best_fail = cand
    if best is None and best_fail is not None:
        res.update(best_fail)
        res["reason"] = (f"TTL lines {best_fail['probe_line']} and {best_fail['adc_line']} match "
                         f"({best_fail['n_matched']} pulse pair(s) in the whole session), but inside the retained interval: "
                         + "; ".join(best_fail["problems"]))
        return res
    if best is None:
        res["reason"] = (f"no TTL line of the ADC stream matches a probe TTL line within {1000 * tolerance_s:g} ms"
                         + (f" (closest: {closest[0]} vs {closest[1]}, median difference {1000 * closest[2]:.3f} ms)"
                            if closest else ""))
        return res
    res.update(best)
    res["status"] = "verified"
    return res


def align_adc_to_probe(adc_info, probe_timestamps_path, probe_crop_start_frame, recording_root,
                       probe_label, tolerance_ms=1.0, max_sync_gap_s=None):
    """
    Select the ADC samples covering the retained probe interval, using both streams' own
    synchronized timestamps, ONLY if a shared clock is established:
      1. probe timestamps valid (finite, not -1, increasing) over the retained part;
      2. ADC timestamps valid and increasing (a leading block of -1 = not yet synchronized is
         allowed and excluded); the ADC covers the whole retained interval (no extrapolation);
      3. shared TTL pulses carry the same timestamps in both streams, paired one-to-one and
         spanning the retained interval (compare_sync_events; max_sync_gap_s -> its max_gap_s);
      4. no repeated / backward ADC sample numbers in the selected range (forward gaps are
         reported: then use the ADC timestamps, not frame / fs);
      5. no timestamp jumps the sample numbers don't explain (scan_stream_clock with fs), in the
         retained probe part (probe sample_numbers.npy, next to its timestamps.npy, is required)
         and in the selected ADC range: e.g. +100 ms with continuous sample numbers -> unavailable.
    Returns a dict; status 'aligned', 'aligned_with_gaps' or 'unavailable' (+ reason).
    For aligned: adc_start_index / adc_end_index (end exclusive) into the ORIGINAL ADC stream,
    probe_t0_s / probe_t_last_s (Open Ephys clock), and the selected ADC times.
    """
    import numpy as np
    from pathlib import Path
    pts = np.load(probe_timestamps_path, mmap_mode="r")
    c = int(probe_crop_start_frame)
    fs_adc = float(adc_info["sampling_rate_hz"])
    out = {"status": "unavailable", "reason": None, "probe_timestamps": str(probe_timestamps_path),
           "adc_timestamps": adc_info["files"]["timestamps.npy"],
           "adc_sample_numbers": adc_info["files"]["sample_numbers.npy"],
           "probe_crop_start_frame": c, "tolerance_ms": float(tolerance_ms)}

    def fail(reason):
        out["status"], out["reason"] = "unavailable", reason
        return out

    if not 0 <= c < len(pts):
        return fail(f"probe crop start frame {c} is outside the probe timestamps (length {len(pts):,})")
    psn_path = Path(probe_timestamps_path).with_name("sample_numbers.npy")
    if not psn_path.is_file():
        raise FileNotFoundError(f"Missing probe sample_numbers.npy (written by Open Ephys >= 0.6): {psn_path}\n"
                                "The ADC alignment needs it to check the probe clock; it does not reconstruct it.")
    psn = np.load(psn_path, mmap_mode="r")
    if len(psn) != len(pts):
        raise ValueError(f"Probe timestamps.npy ({len(pts):,}) and sample_numbers.npy ({len(psn):,}) differ in length: "
                         f"{probe_timestamps_path}")
    out["probe_sample_numbers"] = str(psn_path)
    p_scan = scan_stream_clock(pts, None, c, len(pts), max_backward_s=SYNC_CORRECTION_MAX_S)
    out["probe_scan"] = p_scan
    if not p_scan["ok_clock"]:
        return fail("probe timestamps are not valid over the retained part (non-finite, -1, negative or "
                    f"non-increasing values; first at sample {p_scan['first_bad_timestamp_index']})")
    fs_probe = _stream_rate_from_oebin(recording_root, probe_label)
    p_jump = scan_stream_clock(pts, psn, c, len(pts), max_backward_s=SYNC_CORRECTION_MAX_S, fs=fs_probe)
    out["probe_timing_scan"] = p_jump
    if not p_jump["ok_samples"]:
        return fail(f"probe sample numbers repeat ({p_jump['n_repeated_samples']}) or run backwards "
                    f"({p_jump['n_backward_jumps']}) over the retained part")
    if not p_jump["ok_timing"]:
        return fail(_jump_reason("probe", p_jump))
    t0, t1 = float(pts[c]), float(pts[len(pts) - 1])
    out.update(probe_t0_s=t0, probe_t_last_s=t1)

    ats = np.load(adc_info["files"]["timestamps.npy"], mmap_mode="r")
    asn = np.load(adc_info["files"]["sample_numbers.npy"], mmap_mode="r")
    n = len(ats)
    v0, chunk = 0, 1_000_000
    while v0 < n:                                       # leading samples Open Ephys couldn't synchronize (-1)
        blk = np.asarray(ats[v0:v0 + chunk])
        nz = np.flatnonzero(blk != -1.0)
        if nz.size:
            v0 += int(nz[0])
            break
        v0 += blk.size
    out["adc_leading_unsynchronized_samples"] = int(v0)
    if v0 >= n:
        return fail("every ADC timestamp is -1: Open Ephys never synchronized this stream")
    a_scan = scan_stream_clock(ats, None, v0, n, max_backward_s=SYNC_CORRECTION_MAX_S)
    out["adc_scan"] = a_scan
    if not a_scan["ok_clock"]:
        return fail("ADC timestamps are not valid after the first synchronized sample (non-finite, -1, negative "
                    f"or non-increasing values; first at sample {a_scan['first_bad_timestamp_index']})")
    a_first, a_last = float(ats[v0]), float(ats[n - 1])
    out.update(adc_first_synchronized_s=a_first, adc_last_s=a_last)
    if a_first > t0 or a_last < t1:
        parts = []
        if a_first > t0:
            parts.append(f"starts {a_first - t0:.6f} s after the retained probe interval begins")
        if a_last < t1:
            parts.append(f"ends {t1 - a_last:.6f} s before the retained probe interval ends")
        return fail("insufficient ADC coverage: the synchronized ADC data " + " and ".join(parts)
                    + f" (probe {t0:.6f}-{t1:.6f} s, ADC {a_first:.6f}-{a_last:.6f} s). Nothing is extrapolated.")

    sync = compare_sync_events(recording_root, probe_label, adc_info["adc_label"], tolerance_ms / 1000.0,
                               window=(t0, t1), max_gap_s=max_sync_gap_s)
    out["sync_evidence"] = sync
    if sync["status"] != "verified":
        return fail("shared clock not established: " + sync.get("reason", "no evidence"))

    i0 = _first_index_at_or_after(ats, t0, v0)
    i1 = _last_index_at_or_before(ats, t1, v0) + 1
    s_scan = scan_stream_clock(ats, asn, i0, i1, max_backward_s=SYNC_CORRECTION_MAX_S, fs=fs_adc)
    out["selected_scan"] = s_scan
    if not s_scan["ok_samples"]:
        return fail(f"ADC sample numbers repeat ({s_scan['n_repeated_samples']}) or run backwards "
                    f"({s_scan['n_backward_jumps']}) inside the selected range")
    if not s_scan["ok_timing"]:
        return fail(_jump_reason("ADC", s_scan, offset_note=" (index into the original ADC stream)"))
    out.update(
        status="aligned_with_gaps" if s_scan["n_forward_gaps"] else "aligned",
        reason=None if not s_scan["n_forward_gaps"] else (
            f"{s_scan['n_forward_gaps']} forward gap(s), {s_scan['missing_samples_total']} missing ADC samples: "
            "frame / fs is not elapsed time; use the ADC timestamps"),
        adc_start_index=i0, adc_end_index=i1, adc_n_selected=i1 - i0,
        adc_first_selected_s=float(ats[i0]), adc_last_selected_s=float(ats[i1 - 1]),
        adc_first_selected_minus_probe_t0_ms=1000 * (float(ats[i0]) - t0),
        time_zero="first retained probe sample (probe_t0_s) = Kilosort / cropped-event time zero",
    )
    return out


def _jump_reason(which, scan, offset_note=""):
    """Message for timestamp jumps the sample numbers don't explain (from scan_stream_clock)."""
    ex = scan["first_unexplained_timestamp_jumps"][:3]
    ex_txt = "; ".join(f"step {e['index']:,}->{e['index'] + 1:,}: {1e3 * e['step_s']:.3f} ms, expected "
                       f"{1e3 * e['expected_step_s']:.3f} ms (sample-number step {e['sample_number_step']})" for e in ex)
    return (f"{which} timestamps jump in a way the sample numbers don't explain: "
            f"{scan['n_unexplained_timestamp_jumps']} step(s), largest deviation "
            f"{1e3 * scan['largest_unexplained_jump_s']:.3f} ms (allowed: half a sample + "
            f"{1e3 * scan['jump_tolerance_s']:g} ms); first {ex_txt}{offset_note}. "
            "The clock is not continuous, so the ADC is not aligned.")


def _stream_rate_from_oebin(recording_root, folder_name):
    """Sampling rate of one continuous stream from structure.oebin (raises if not found)."""
    import json
    from pathlib import Path
    p = Path(recording_root) / "structure.oebin"
    d = json.loads(p.read_text(encoding="utf-8"))
    for c in d.get("continuous", []):
        if c.get("folder_name", "").rstrip("/") == folder_name:
            return float(c["sample_rate"])
    raise ValueError(f"Stream {folder_name!r} not listed in {p}; cannot get its sampling rate.")


def _first_index_at_or_after(ts, t, lo=0, window=100_000):
    """First index i >= lo with ts[i] >= t, exact even with sub-sample backward corrections."""
    import numpy as np
    k = lo + int(np.searchsorted(ts[lo:], t, side="left"))
    a = max(lo, k - window)
    seg = np.asarray(ts[a:min(len(ts), k + window)])
    hit = np.flatnonzero(seg >= t)
    return a + int(hit[0]) if hit.size else len(ts)


def _last_index_at_or_before(ts, t, lo=0, window=100_000):
    """Last index i >= lo with ts[i] <= t (exact, as above)."""
    import numpy as np
    k = lo + int(np.searchsorted(ts[lo:], t, side="right"))
    b = min(len(ts), k + window)
    a = max(lo, k - window)
    seg = np.asarray(ts[a:b])
    hit = np.flatnonzero(seg <= t)
    return a + int(hit[-1]) if hit.size else lo - 1


def adc_cropped_times(alignment, start_frame=0, end_frame=None):
    """
    Measured times (s) of cropped-ADC frames [start_frame, end_frame), relative to the first
    retained probe sample (the zero of Kilosort spike_times / fs and of the cropped events).
    Read from the ADC's own timestamps.npy; needs status 'aligned' or 'aligned_with_gaps'.
    """
    import numpy as np
    if alignment.get("status") not in ("aligned", "aligned_with_gaps"):
        raise ValueError(f"ADC alignment is {alignment.get('status')}: {alignment.get('reason')}")
    i0, i1 = int(alignment["adc_start_index"]), int(alignment["adc_end_index"])
    end_frame = i1 - i0 if end_frame is None else int(end_frame)
    if not 0 <= start_frame <= end_frame <= i1 - i0:
        raise ValueError(f"Frames {start_frame}-{end_frame} outside the cropped ADC (0-{i1 - i0}).")
    ts = np.load(alignment["adc_timestamps"], mmap_mode="r")
    return np.asarray(ts[i0 + start_frame:i0 + end_frame], dtype=np.float64) - float(alignment["probe_t0_s"])


# Set True (e.g. nam.STRICT_SOURCE_HASH = True before S3) to also store a full SHA-256 of the
# original timestamps.npy / sample_numbers.npy; S4/A1 then re-read both files completely
# (several GB for long recordings: tens of seconds).
STRICT_SOURCE_HASH = False


def file_sha256(path, block=64 * 1024 * 1024):
    """Streaming SHA-256 of a whole file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_file_identity(path, sample_bytes=4 * 1024 * 1024, full_hash=None):
    """
    Identity of an acquisition file, recorded by S3 and rechecked by S4 / A1 (added 2026-10-30):
    size, modification time (ns) and SHA-1 of its first and last 4 MB. Detects replaced or
    rewritten files (any write changes the modification time); a change that keeps size and
    modification time and lies outside the sampled bytes would not be detected, unless
    full_hash (default STRICT_SOURCE_HASH) adds a full SHA-256 of the file.
    """
    import hashlib, os
    from pathlib import Path
    path = Path(path)
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(sample_bytes))
        if st.st_size > 2 * sample_bytes:
            f.seek(st.st_size - sample_bytes)
            h.update(f.read(sample_bytes))
        elif st.st_size > sample_bytes:
            h.update(f.read())
    ident = {"path": str(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
             "sha1_head_tail": h.hexdigest()}
    if STRICT_SOURCE_HASH if full_hash is None else full_hash:
        ident["sha256_full"] = file_sha256(path)
    return ident


def verify_trimmed_timestamps(preprocessed_folder, si_save_path):
    """
    Check S3's outputs before they are used (S4, A1) (added 2026-10-30):
      - timestamps_trimmed.npy, sample_numbers_trimmed.npy, their settings file and
        timestamp_gap_report.json exist (else: run S3);
      - they were READ from Open Ephys files (settings 's3_provenance', report
        'sample_numbers_inferred' = false); outputs of older versions that inferred sample
        numbers, or that carry no provenance, are refused (they are not deleted);
      - the crop/channel settings equal those of the current S2 folder, and the lengths equal
        its frame count.
    Returns (settings, gap_report). Raises with what to rerun.
    """
    import json
    import numpy as np
    from pathlib import Path
    pre, s2 = Path(preprocessed_folder), Path(si_save_path)
    ts_path = pre / "timestamps_trimmed.npy"
    sn_path = pre / "sample_numbers_trimmed.npy"
    stamp_path = ts_path.with_name(ts_path.name + ".settings.json")
    report_path = pre / "timestamp_gap_report.json"
    s2_stamp_path = s2 / SETTINGS_STAMP_NAME
    missing = [str(p) for p in (ts_path, sn_path, stamp_path, report_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing S3 output(s):\n  " + "\n  ".join(missing) + "\nRun S3 first.")
    if not s2_stamp_path.exists():
        raise FileNotFoundError(f"The spike-band folder has no settings file: {s2_stamp_path}\nRerun S2.")
    stamp = json.loads(stamp_path.read_text())
    report = json.loads(report_path.read_text())
    prov = stamp.get("s3_provenance")
    if report.get("sample_numbers_inferred") is not False or not prov or prov.get("sample_numbers") != "read":
        why = ("its sample numbers were INFERRED from timestamps by an older notebook version"
               if report.get("sample_numbers_inferred") else "it records no provenance (older notebook version)")
        raise ValueError(
            f"The trimmed timestamps in {pre} can't be used: {why}.\n"
            "Rerun S3 (it needs the recording's own sample_numbers.npy). The old files were not deleted; "
            "S3 replaces them."
        )
    ident = prov.get("source_identity")
    if not ident or not all(k in ident for k in ("timestamps", "sample_numbers")):
        raise ValueError(f"The trimmed timestamps in {pre} don't record their source files' identity "
                         "(older S3 version). Rerun S3.")
    for key, label in (("timestamps", "timestamps.npy"), ("sample_numbers", "sample_numbers.npy")):
        src = Path(ident[key]["path"])
        if not src.is_file():
            raise FileNotFoundError(f"The original {label} that S3 read is missing: {src}\n"
                                    "Restore it from the original recording; it is never recreated.")
        now = source_file_identity(src, full_hash="sha256_full" in ident[key])
        changed = [k for k in ("size", "mtime_ns", "sha1_head_tail", "sha256_full")
                   if k in ident[key] and now.get(k) != ident[key].get(k)]
        if changed:
            raise ValueError(
                f"The original {label} has changed since S3 read it ({', '.join(changed)} differ): {src}\n"
                "The trimmed timestamps may not match it any more. Rerun S3 (and the cells after it)."
            )
    # S3's own outputs: full SHA-256 of the two arrays and the report, recorded by S3
    out_ids = stamp.get("s3_output_sha256")
    if not out_ids:
        raise ValueError(f"The S3 settings in {pre} don't record their outputs' checksums (older S3 version). Rerun S3.")
    for path in (ts_path, sn_path, report_path):
        if file_sha256(path) != out_ids.get(path.name):
            raise ValueError(f"{path} has changed since S3 wrote it (checksum differs). Rerun S3.")
    if report.get("clock_ok_for_sorting") is not True:
        raise ValueError(
            f"timestamp_gap_report.json in {pre} doesn't confirm a usable clock "
            + ("(older S3 version: rerun S3)." if "clock_ok_for_sorting" not in report else
               f"(NaN/inf timestamps {report.get('n_nonfinite_timestamps')}, repeated samples "
               f"{report.get('n_repeated_samples')}, backward samples {report.get('n_backward_jumps')}, "
               f"backward timestamp steps {report.get('n_timestamp_backward_steps')}, unexplained timestamp jumps "
               f"{report.get('n_unexplained_timestamp_jumps')}).")
        )
    s2_stamp = json.loads(s2_stamp_path.read_text())
    diffs = [k for k in SHARED_SETTINGS_KEYS if stamp.get(k) != s2_stamp.get(k)]
    if diffs:
        raise ValueError(
            "The trimmed timestamps (S3) and the spike-band folder (S2) come from different preprocessing:\n"
            + "\n".join(f"  {k}: S2 = {s2_stamp.get(k)!r}, S3 = {stamp.get(k)!r}" for k in diffs)
            + "\nRerun S3 after S2."
        )
    n_ts = len(np.load(ts_path, mmap_mode="r"))
    n_sn = len(np.load(sn_path, mmap_mode="r"))
    n_s2 = int(s2_stamp["raw_n_frames"])
    if not n_ts == n_sn == n_s2:
        raise ValueError(f"Lengths differ: timestamps_trimmed {n_ts:,}, sample_numbers_trimmed {n_sn:,}, "
                         f"S2 {n_s2:,} frames. Rerun S3.")
    return stamp, report



def resolve_probe_crop(config, namespace=None):
    """
    The probe crop of the recording in `config` (added 2026-10-30): read from that recording's
    custom_analytics/recording_crop_timing.npz (written by 1.1), which is the source of truth.
    If `namespace` (e.g. globals()) holds PROBE_CROP_START_FRAME or shared_settings from an
    earlier 1.1 run, they must agree with it (same crop, same recording); otherwise this raises,
    so a crop left in memory from another recording or setting is never used.
    Returns (crop_start_frame, crop_start_s).
    """
    import numpy as np
    from pathlib import Path
    recording_root = (Path(config["base_path"]) / config["record_node"] / config["experiment_label"]
                      / config["recording_label"])
    path = Path(config["data_folder"]) / "custom_analytics" / "recording_crop_timing.npz"
    if not path.exists():
        raise FileNotFoundError(f"Crop metadata not found: {path}\nRun 1.1 for this recording.")
    z = np.load(path)
    missing = [k for k in ("recording_root", "probe_n_frames_full", "remove_first_minutes") if k not in z.files]
    if missing:
        raise ValueError(f"{path} was written by an older notebook version (no {', '.join(missing)}). Rerun 1.1.")
    frame, t = int(z["probe_crop_start_frame"]), float(z["probe_crop_start_s"])
    saved_minutes = float(z["remove_first_minutes"])
    if Path(str(z["recording_root"])) != recording_root:
        raise ValueError(f"{path} belongs to {z['recording_root']}, not {recording_root}. Rerun 1.1.")
    ns = namespace or {}
    requested = [("REMOVE_FIRST_MINUTES in memory (0.2)", ns.get("REMOVE_FIRST_MINUTES")),
                 ("remove_first_minutes in the configuration", config.get("remove_first_minutes"))]
    ss_ = ns.get("shared_settings")
    if isinstance(ss_, dict):
        requested.append(("the preprocessing in memory (1.1)", ss_.get("remove_first_minutes")))
    for label, value in requested:
        if value is not None and abs(float(value) - saved_minutes) > 1e-12:
            raise ValueError(
                f"{label} is {float(value):g} min, but the saved crop ({path}) was made with {saved_minutes:g} min. "
                "Rerun 1.1 (and the cells after it) for the current setting."
            )
    if "PROBE_CROP_START_FRAME" in ns and int(ns["PROBE_CROP_START_FRAME"]) != frame:
        raise ValueError(
            f"The crop in memory (frame {int(ns['PROBE_CROP_START_FRAME']):,}) differs from this recording's saved "
            f"crop (frame {frame:,}, {path}). It probably comes from another recording or setting: rerun 1.1 "
            "for the current configuration."
        )
    ss = ns.get("shared_settings")
    if isinstance(ss, dict) and ss.get("recording_root") and Path(ss["recording_root"]) != recording_root:
        raise ValueError(
            f"The preprocessing in memory (1.1) is for {ss['recording_root']}, but the configuration is for "
            f"{recording_root}. Rerun 1.1 for the current configuration."
        )
    return frame, t



# ============================================================
# MULTI-SOURCE ADC SELECTION (added 2026-10-31): ADC_MODE 'off' | 'nidaq' | 'onebox' | 'both'
# ============================================================

ADC_MODES = {"off": (), "nidaq": ("nidaq",), "onebox": ("onebox",), "both": ("nidaq", "onebox")}
ADC_SOURCE_FIELDS = ("label", "channel_ids", "stream_name", "unit_if_not_stated", "record_node")


def selected_adc_keys(adc_mode):
    """The source keys an ADC_MODE selects, in a fixed order. Anything else raises (no fallback).
    'legacy' = a configuration written with the older single-ADC keys (one source, key 'adc')."""
    if adc_mode == "legacy":
        return ["adc"]
    if not isinstance(adc_mode, str) or adc_mode not in ADC_MODES:
        raise ValueError(f"ADC_MODE must be one of {list(ADC_MODES)}; got {adc_mode!r}.")
    return list(ADC_MODES[adc_mode])


def resolve_adc_sources(recording_root, experiment_label, record_node, adc_mode, adc_sources, probe_label):
    """
    Check and resolve every source the ADC_MODE selects (only those): its folder and Open Ephys
    files must exist and the reader must find its stream (resolve_adc_stream). Sources must be in
    the probe's own Record Node / recording folder: the shared-clock check compares TTL events
    within one recording, so a source in another Record Node is refused (not compared against
    the wrong folder). Duplicate folders or stream IDs across sources are refused.
    Returns a list of dicts: key, label, folder, record_node, stream_name, stream_id,
    channel_ids, unit_if_not_stated.
    """
    from pathlib import Path
    recording_root = Path(recording_root)
    keys = selected_adc_keys(adc_mode)
    if adc_mode == "legacy":
        raise ValueError("'legacy' is only used for configurations read from older files; choose an ADC_MODE.")
    out = []
    for key in keys:
        spec = (adc_sources or {}).get(key)
        if not isinstance(spec, dict) or not spec.get("label"):
            raise ValueError(f"ADC_MODE {adc_mode!r} selects '{key}', but ADC_SOURCES['{key}']['label'] is not set.")
        unknown = sorted(set(spec) - set(ADC_SOURCE_FIELDS))
        if unknown:
            raise ValueError(f"ADC_SOURCES['{key}'] has unknown field(s) {unknown}; allowed: {list(ADC_SOURCE_FIELDS)}.")
        rn = spec.get("record_node")
        if rn is not None and rn != record_node:
            raise ValueError(
                f"ADC_SOURCES['{key}'] is in {rn!r}, the probe in {record_node!r}. Sources in another Record Node "
                "are not supported: the shared-clock check compares TTL events within one recording folder."
            )
        unit = spec.get("unit_if_not_stated")
        if unit is not None and unit not in _VOLTAGE_UNITS_TO_UV:
            raise ValueError(f"ADC_SOURCES['{key}']['unit_if_not_stated'] must be one of "
                             f"{list(_VOLTAGE_UNITS_TO_UV)} or None; got {unit!r}.")
        ch = spec.get("channel_ids")
        if ch is not None and (isinstance(ch, str) or not len(ch)):
            raise ValueError(f"ADC_SOURCES['{key}']['channel_ids'] must be None or a non-empty list of IDs.")
        try:
            name, sid, _ = resolve_adc_stream(recording_root, experiment_label, record_node, spec["label"],
                                              stream_name=spec.get("stream_name"), probe_label=probe_label)
        except (FileNotFoundError, ValueError) as exc:
            raise type(exc)(f"ADC source '{key}' ({spec['label']}): {exc}") from None
        out.append({"key": key, "label": spec["label"], "folder": str(recording_root / "continuous" / spec["label"]),
                    "record_node": record_node, "stream_name": name, "stream_id": str(sid),
                    "channel_ids": [str(c) for c in ch] if ch is not None else None, "unit_if_not_stated": unit})
    for field in ("label", "stream_id"):
        vals = [src[field] for src in out]
        if len(set(vals)) != len(vals):
            raise ValueError(f"Two ADC sources resolve to the same {field}: {vals}. Each source needs its own stream.")
    return out


def load_adc_source(config, source):
    """
    Load ONE selected source (a dict from config['adc_sources']) with all of load_adc_recording()'s
    checks: same stream as resolved in 0.2, required files, lengths, own rate, units from
    structure.oebin or the source's own unit_if_not_stated. Returns (recording, info).
    """
    view = {**config, "load_adc": True, "ADC_label": source["label"],
            "adc_stream_name": source.get("stream_name"), "adc_stream_id": source.get("stream_id")}
    if source.get("record_node") not in (None, config["record_node"]):
        raise ValueError(f"ADC source '{source.get('key')}' is in {source['record_node']}, the probe in "
                         f"{config['record_node']}: not supported.")
    rec, info = load_adc_recording(view, channel_ids=source.get("channel_ids"),
                                   unit_if_not_stated=source.get("unit_if_not_stated"))
    info["source_key"] = source.get("key")
    return rec, info
