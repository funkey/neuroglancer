# @license
# Copyright 2016 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import numpy as np

from .chunks import encode_jpeg, encode_npz, encode_raw
from .token import make_random_token
from scipy.ndimage import zoom

# scale dependent:
#   shape (in voxels)
#   offset (in voxels)
#   voxel_size
# new:
#   scales

def scale_tuple(t, f):
    return tuple([x*f for x in t])

class ServedVolume(object):
  def __init__(self,
               data,
               offset=(0, 0, 0),
               voxel_size=(1, 1, 1),
               encoding='npz',
               chunk_data_sizes=None,
               volume_type=None,
               scales = [1, 2, 4, 8]):
    """Initializes a ServedVolume.

    @param data 3-d [z, y, x] array or 4-d [channel, z, y, x] array.
    """
    self.token = make_random_token()
    self.scales = scales
    self.scale_key_to_index = { self.scale_key(s) : i for (i,s) in enumerate(self.scales) }
    if len(data.shape) == 3:
      self.num_channels = 1
      self.shape = { s: scale_tuple(data.shape[::-1], 1.0/s) for s in self.scales }
    else:
      if len(data.shape) != 4:
        raise ValueError('data array must be 3- or 4-dimensional.')
      self.num_channels = data.shape[0]
      self.shape = { s: scale_tuple(data.shape[1:][::-1], 1.0/s) for s in self.scales }

    self.data = data
    self.voxel_size = { s: scale_tuple(tuple(float(x) for x in voxel_size), s) for s in self.scales }
    self.offset = { s: scale_tuple(offset, s) for s in self.scales }
    self.data_type = data.dtype.name
    self.encoding = encoding
    if chunk_data_sizes is not None:
      arr = np.array(chunk_data_sizes)
      if (len(arr.shape) != 2 or arr.shape[1] != 3 or np.any(arr < 1) or
          np.any(np.cast[int](arr) != arr)):
        raise ValueError(
            'chunk_data_sizes must be a sequence of 3-element non-negative integers')
    self.chunk_data_sizes = chunk_data_sizes
    if volume_type is None:
      if self.num_channels == 1 and (self.data_type == 'uint16' or
                                     self.data_type == 'uint32' or
                                     self.data_type == 'uint64'):
        volume_type = 'segmentation'
      else:
        volume_type = 'image'
    self.volume_type = volume_type

  def info(self):
    upper_voxel_bound = { s : tuple(np.array(self.offset[s]) + np.array(self.shape[s])) for s in self.scales }
    info = dict(volumeType=self.volume_type,
                dataType=self.data_type,
                encoding=self.encoding,
                numChannels=self.num_channels,
                scales=[
                    dict(volume_key=self.token,
                         scale_key=self.scale_key(s),
                         lowerVoxelBound=self.offset[s],
                         upperVoxelBound=upper_voxel_bound[s],
                         voxelSize=self.voxel_size[s])
                    for s in self.scales
                ])
    if self.chunk_data_sizes is not None:
      info['chunkDataSizes'] = self.chunk_data_sizes
    return info

  def scale_key(self, scale):
    return str(scale)

  def get_encoded_subvolume(self, data_format, start, end, scale_key=None):
    scale = 1
    if scale_key is not None:
        scale = self.scales[self.scale_key_to_index[scale_key]]
    offset = self.offset[scale]
    shape = self.shape[scale]
    print("offset: " + str(offset))
    print("shape : " + str(shape))
    print("start : " + str(start))
    print("end   : " + str(end))
    print("scale : " + str(scale))
    for i in xrange(3):
      if end[i] < start[i] or offset[i] > start[i] or end[i] - offset[i] > shape[i]:
        raise ValueError('Out of bounds data request.')

    indexing_expr = tuple(np.s_[(start[i] - offset[i])*scale:(end[i] - offset[i])*scale] for i in (2,1,0))
    print("indexing expression: " + str(indexing_expr))
    if len(self.data.shape) == 3:
      subvol = self.data[indexing_expr]
      if scale != 1:
        print("subvol was: " + str(subvol.shape))
        subvol = zoom(subvol, 1.0/scale)
        print("subvol is : " + str(subvol.shape))
    else:
      subvol = self.data[(np.s_[:],) + indexing_expr]
      if scale != 1:
        print("subvol was: " + str(subvol.shape))
        subvol = np.array([ zoom(subvol[c], 1.0/scale) for c in xrange(subvol.shape[0]) ])
        print("subvol is : " + str(subvol.shape))
    content_type = 'application/octet-stream'
    if data_format == 'jpeg':
      data = encode_jpeg(subvol)
      content_type = 'image/jpeg'
    elif data_format == 'npz':
      data = encode_npz(subvol)
    elif data_format == 'raw':
      data = encode_raw(subvol)
    else:
      raise ValueError('Invalid data format requested.')
    return data, content_type

  def get_object_mesh(self, object_id):
    raise ValueError('Meshes not yet supported.')
