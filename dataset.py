import ast
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from natsort import natsorted
from torch.nn import functional as F
from torch.utils.data import Dataset

# Helper functions
rand_range = lambda amin, amax: amin + (amax - amin) * np.random.rand()

val_files = ["1_6", "2_4", "4_4", "6_2", "7_4", "9_1", "10_3", "11_2", "12_3"]

def get_index(file_lens, index):
    file_lens_cumsum = np.cumsum(np.array(file_lens))
    file_id = np.searchsorted(file_lens_cumsum, index, side='right')
    sample_id = index - file_lens_cumsum[file_id - 1] if file_id > 0 else index
    return file_id, sample_id

def txt_to_npy(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(ast.literal_eval(line.strip()))
    return np.array(data)

def h5_to_npy(file_path, name):
    with h5py.File(file_path, 'r') as file:
        npy_data = file[name][:]
    return npy_data

def bilinear_interp(x, scale, x_max):
    if scale == 1:
        return x, x, torch.ones_like(x), torch.zeros_like(x)
    xd1 = (x % scale) / scale
    xd = 1 - xd1
    x = (x / scale).long().clamp(0, x_max)
    x1 = (x + 1).clamp(0, x_max)
    return x, x1, xd, xd1

def compute_density(events, num_bins, lookback=10000, device='cpu'):
    """Compute event density based on a lookback window."""
    timestamps = events[-1]
    density = []
    bin_edges = torch.linspace(0, timestamps.max().item(), num_bins + 1, device=device)
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        lookback_start = max(0, bin_start - lookback)
        num_events = torch.sum((timestamps >= lookback_start) & (timestamps < bin_start))
        density.append(num_events.item() / (bin_start - lookback_start + 1e-6))
    return torch.tensor(density, device=device)

def adjust_bin_sizes(density, base_size=10000, beta=1.0, min_size=5000, max_size=20000):
    """Adjust bin sizes inversely proportional to density with constraints."""
    density = density.float()
    density_min, density_max = density.min(), density.max()
    if density_max == density_min:
        return torch.full_like(density, base_size, dtype=torch.float)
    normalized_density = (density - density_min) / (density_max - density_min)
    bin_sizes = base_size * (1 + beta * normalized_density).reciprocal()
    return bin_sizes.clamp(min_size, max_size)

def events_to_single_frame(events, size, spatial_downsample, mode='bilinear'):
    """Convert events within a single bin to a frame with spatial interpolation."""
    height, width = size
    p, x, y, t = events
    frame = torch.zeros([2, height, width], dtype=torch.float, device=events.device)
    
    if mode == 'nearest':
        p = p.round().long()
        x = (x / spatial_downsample[0]).round().long().clamp(0, width - 1)
        y = (y / spatial_downsample[1]).round().long().clamp(0, height - 1)
        frame.index_put_((p, y, x), torch.ones_like(p, dtype=torch.float), accumulate=True)
    else:  # 'bilinear' or 'causal_linear'
        x, x1, xd, xd1 = bilinear_interp(x, spatial_downsample[0], width - 1)
        y, y1, yd, yd1 = bilinear_interp(y, spatial_downsample[1], height - 1)
        p = p.long().repeat(4)
        x = torch.cat([x, x, x1, x1])
        y = torch.cat([y, y1, y, y1])
        weights = torch.cat([xd * yd, xd * yd1, xd1 * yd, xd1 * yd1])
        frame.index_put_((p, y, x), weights, accumulate=True)
    return frame

def events_to_frames(events, size, num_frames, spatial_downsample, bin_sizes, mode='bilinear'):
    """Convert events to frames with adaptive temporal binning and resampling."""
    height, width = size
    p, x, y, t = events
    bin_edges = torch.cat([torch.zeros(1, device=events.device), bin_sizes.cumsum(0)])
    
    adaptive_frames = []
    for i in range(len(bin_edges) - 1):
        mask = (t >= bin_edges[i]) & (t < bin_edges[i + 1])
        events_in_bin = events[:, mask]
        if events_in_bin.shape[1] == 0:
            frame = torch.zeros(2, height, width, dtype=torch.float, device=events.device)
        else:
            frame = events_to_single_frame(events_in_bin, size, spatial_downsample, mode)
        adaptive_frames.append(frame)
    adaptive_frames = torch.stack(adaptive_frames)  # (T_adaptive, 2, H, W)
    
    # Resample to the fixed number of frames
    adaptive_t = adaptive_frames.shape[0]
    if adaptive_t == num_frames:
        events_frames = adaptive_frames
    else:
        events_frames = F.interpolate(
            adaptive_frames.permute(1, 2, 3, 0).unsqueeze(0),  # (1, 2, H, W, T_adaptive)
            size=(num_frames,), mode='linear', align_corners=False
        ).squeeze(0).permute(4, 0, 1, 2)  # (num_frames, 2, H, W)
    
    return events_frames.type_as(events)

# Assuming EventRandomAffine class remains unchanged from the original
class EventRandomAffine:
    def __init__(self, size, augment_flag=True):
        self.size = size
        self.augment_flag = augment_flag
    
    def __call__(self, event, label):
        # Placeholder for spatial affine transformation logic
        # This should be the original implementation
        return event, label[:, :2], label[:, 2]

class EyeTrackingDataset(Dataset):
    def __init__(self, 
                 root_path, 
                 mode='train', 
                 device='cpu', 
                 time_window=10000, 
                 frames_per_segment=50, 
                 spatial_downsample=(5, 5), 
                 events_interpolation='bilinear', 
                 spatial_affine=True, 
                 temporal_flip=True, 
                 temporal_scale=True, 
                 temporal_shift=True,
                 test_on_val=False):
        self.mode = mode
        self.time_window = time_window
        self.frames_per_segment = frames_per_segment
        self.time_window_per_segment = time_window * frames_per_segment
        self.spatial_downsample = spatial_downsample
        self.events_interpolation = events_interpolation
        assert time_window == 10000
        
        self.temporal_flip = temporal_flip
        self.temporal_scale = temporal_scale
        self.temporal_shift = temporal_shift
        self.test_on_val = test_on_val
        
        root_path = Path(root_path)
        if mode in ['train', 'val']:
            base_path = root_path / 'train'
        elif mode == 'test':
            base_path = root_path / ('train' if test_on_val else 'test')
        else:
            raise ValueError("Invalid mode. Must be 'train', 'val', or 'test'.")
        
        self.events, self.labels = [], []
        self.num_frames_list, self.num_segments_list = [], []
        
        dir_paths = natsorted(base_path.glob('*'))
        if mode == 'train':
            dir_paths = [d for d in dir_paths if d.name not in val_files]
        elif mode == 'val' or (mode == 'test' and test_on_val):
            dir_paths = [d for d in dir_paths if d.name in val_files]

        for dir_path in dir_paths:
            assert dir_path.is_dir()
            data_path = dir_path / f'{dir_path.name}.h5'
            label_path = dir_path / ('label.txt' if (mode != 'test' or test_on_val) else 'label_zeros.txt')
            
            event, label = h5_to_npy(data_path, 'events'), txt_to_npy(label_path)
            
            num_frames = label.shape[0]
            self.num_frames_list.append(num_frames)
            self.num_segments_list.append(num_frames // frames_per_segment)
            
            final_t = num_frames * time_window
            final_ind = np.searchsorted(event['t'], final_t, 'left')
            event = event[:final_ind]
            
            label = torch.tensor(label, dtype=torch.float, device=device)
            event = np.stack([event['p'].astype('float32'), event['x'].astype('float32'), 
                            event['y'].astype('float32'), event['t'].astype('float32')], axis=0)
            event = torch.tensor(event, dtype=torch.float, device=device)
            
            self.events.append(event)
            self.labels.append(label)
        
        self.total_segments = sum(self.num_segments_list)
        self.augment = EventRandomAffine((480, 640), augment_flag=(mode == 'train' and spatial_affine))
    
    def __len__(self):
        return len(self.events) if self.mode == 'test' else self.total_segments
    
    def _process_data(self, event, label, index=None):
        event, center, close = self.augment(event, label)
        num_frames = self.frames_per_segment if self.mode != 'test' else self.num_frames_list[index]
        
        # Adaptive temporal binning
        event_density = compute_density(event, num_bins=num_frames, lookback=10000, device=event.device)
        bin_sizes = adjust_bin_sizes(event_density, base_size=10000, beta=1.0, min_size=5000, max_size=20000)
        
        event = events_to_frames(event, 
                                (480 // self.spatial_downsample[1], 640 // self.spatial_downsample[0]), 
                                num_frames, self.spatial_downsample, bin_sizes, 
                                mode=self.events_interpolation)
        
        if self.mode == 'train' and self.temporal_flip and np.random.rand() > 0.5:
            event = event.flip(0).flip(1)  # (T, C, H, W)
            center = center.flip(-1)
            close = close.flip(-1)
        
        return event.moveaxis(0, 1), center, 1 - close
    
    def __getitem__(self, index):
        if self.mode == 'test':
            event, label = self.events[index], self.labels[index]
            return self._process_data(event, label, index)
        
        file_id, segment_id = get_index(self.num_segments_list, index)
        event, label = self.events[file_id], self.labels[file_id]
        
        start_t = segment_id * self.time_window * self.frames_per_segment
        end_t = start_t + self.time_window * self.frames_per_segment
        
        max_offset = round(self.time_window_per_segment * 0.1)
        if self.mode == 'train' and self.temporal_shift and start_t >= max_offset:
            offset = np.random.rand() * max_offset
            start_t -= offset
            end_t -= offset
        else:
            offset = 0
        
        num_frames = self.num_frames_list[file_id]
        event = event.clone()
        if self.mode == 'train' and self.temporal_scale and end_t < (num_frames * self.time_window * 0.8):
            scale_factor = float(rand_range(0.8, 1.2))
            event[-1] *= scale_factor
        else:
            scale_factor = 1
        
        start_ind = torch.searchsorted(event[-1], start_t, side='left')
        end_ind = torch.searchsorted(event[-1], end_t, side='left')
        
        event_segment = event[:, start_ind.item():end_ind.item()]
        event_segment[-1] -= start_t
        
        start_label_id = segment_id * self.frames_per_segment
        end_label_id = (segment_id + 1) * self.frames_per_segment
        
        label_numpy = label.cpu().numpy()
        num_frame = label_numpy.shape[0]
        arange = np.arange(0, num_frame)
        label_offset = offset / self.time_window
        interp_range = np.linspace(
            (start_label_id - label_offset) / scale_factor, 
            (end_label_id - label_offset - 1) / scale_factor, 
            self.frames_per_segment, 
        )
        x_interp = np.interp(interp_range, arange, label_numpy[:, 0])
        y_interp = np.interp(interp_range, arange, label_numpy[:, 1])
        closeness = label_numpy[start_label_id:end_label_id, -1]
        label_segment = torch.tensor(np.stack([x_interp, y_interp, closeness], axis=1)).type_as(label)
        
        return self._process_data(event_segment, label_segment)