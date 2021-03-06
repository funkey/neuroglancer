/**
 * @license
 * Copyright 2016 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {ChunkManager} from 'neuroglancer/chunk_manager/frontend';
import {registerDataSourceFactory} from 'neuroglancer/datasource/factory';
import {MeshSourceParameters, VolumeChunkEncoding, VolumeChunkSourceParameters} from 'neuroglancer/datasource/precomputed/base';
import {defineParameterizedMeshSource} from 'neuroglancer/mesh/frontend';
import {DataType, VolumeChunkSpecification, VolumeType} from 'neuroglancer/sliceview/base';
import {MultiscaleVolumeChunkSource as GenericMultiscaleVolumeChunkSource, defineParameterizedVolumeChunkSource} from 'neuroglancer/sliceview/frontend';
import {Vec3, vec3} from 'neuroglancer/util/geom';
import {openShardedHttpRequest, parseSpecialUrl, sendHttpRequest} from 'neuroglancer/util/http_request';
import {parseArray, parseFixedLengthArray, parseIntVec, stableStringify, verifyEnumString, verifyFinitePositiveFloat, verifyObject, verifyObjectProperty, verifyOptionalString, verifyPositiveInt, verifyString} from 'neuroglancer/util/json';

const VolumeChunkSource = defineParameterizedVolumeChunkSource(VolumeChunkSourceParameters);
const MeshSource = defineParameterizedMeshSource(MeshSourceParameters);

class ScaleInfo {
  key: string;
  encoding: VolumeChunkEncoding;
  resolution: Vec3;
  voxelOffset: Vec3;
  size: Vec3;
  chunkSizes: Vec3[];
  compressedSegmentationBlockSize: Vec3|undefined;
  constructor(obj: any) {
    verifyObject(obj);
    this.resolution = verifyObjectProperty(
        obj, 'resolution', x => parseFixedLengthArray(vec3.create(), x, verifyFinitePositiveFloat));
    this.voxelOffset =
        verifyObjectProperty(obj, 'voxel_offset', x => parseIntVec(vec3.create(), x)),
    this.size = verifyObjectProperty(
        obj, 'size', x => parseFixedLengthArray(vec3.create(), x, verifyPositiveInt));
    this.chunkSizes = verifyObjectProperty(
        obj, 'chunk_sizes',
        x => parseArray(x, y => parseFixedLengthArray(vec3.create(), y, verifyPositiveInt)));
    if (this.chunkSizes.length === 0) {
      throw new Error('No chunk sizes specified.');
    }
    let encoding = this.encoding =
        verifyObjectProperty(obj, 'encoding', x => verifyEnumString(x, VolumeChunkEncoding));
    if (encoding === VolumeChunkEncoding.COMPRESSED_SEGMENTATION) {
      this.compressedSegmentationBlockSize = verifyObjectProperty(
          obj, 'compressed_segmentation_block_size',
          x => parseFixedLengthArray(vec3.create(), x, verifyPositiveInt));
    }
    this.key = verifyObjectProperty(obj, 'key', verifyString);
  }
};

export class MultiscaleVolumeChunkSource implements GenericMultiscaleVolumeChunkSource {
  dataType: DataType;
  numChannels: number;
  volumeType: VolumeType;
  mesh: string|undefined;
  scales: ScaleInfo[];

  getMeshSource(chunkManager: ChunkManager) {
    let {mesh} = this;
    if (mesh === undefined) {
      return null;
    }
    return getShardedMeshSource(
        chunkManager, {baseUrls: this.baseUrls, path: `${this.path}/${mesh}`, lod: 0});
  }

  constructor(public baseUrls: string[], public path: string, private obj: any) {
    verifyObject(obj);
    this.dataType = verifyObjectProperty(obj, 'data_type', x => verifyEnumString(x, DataType));
    this.numChannels = verifyObjectProperty(obj, 'num_channels', verifyPositiveInt);
    this.volumeType = verifyObjectProperty(obj, 'type', x => verifyEnumString(x, VolumeType));
    this.mesh = verifyObjectProperty(obj, 'mesh', verifyOptionalString);
    this.scales = verifyObjectProperty(obj, 'scales', x => parseArray(x, y => new ScaleInfo(y)));
  }

  getSources(chunkManager: ChunkManager) {
    return this.scales.map(scaleInfo => {
      return VolumeChunkSpecification
          .getDefaults({
            voxelSize: scaleInfo.resolution,
            dataType: this.dataType,
            numChannels: this.numChannels,
            chunkLayoutOffset:
                vec3.multiply(vec3.create(), scaleInfo.resolution, scaleInfo.voxelOffset),
            upperVoxelBound: scaleInfo.size,
            volumeType: this.volumeType,
            chunkDataSizes: scaleInfo.chunkSizes,
            baseVoxelOffset: scaleInfo.voxelOffset,
            compressedSegmentationBlockSize: scaleInfo.compressedSegmentationBlockSize
          })
          .map(spec => VolumeChunkSource.get(chunkManager, spec, {
            'baseUrls': this.baseUrls,
            'path': `${this.path}/${scaleInfo.key}`,
            'encoding': scaleInfo.encoding
          }));
    });
  }
};

export function getShardedMeshSource(chunkManager: ChunkManager, parameters: MeshSourceParameters) {
  return MeshSource.get(chunkManager, parameters);
}

export function getMeshSource(chunkManager: ChunkManager, url: string, lod: number) {
  const [baseUrls, path] = parseSpecialUrl(url);
  return getShardedMeshSource(chunkManager, {baseUrls, path, lod});
}

let existingVolumes = new Map<string, Promise<MultiscaleVolumeChunkSource>>();
export function getShardedVolume(baseUrls: string[], path: string) {
  let fullKey = stableStringify({'baseUrls': baseUrls, 'path': path});
  let existingResult = existingVolumes.get(fullKey);
  if (existingResult !== undefined) {
    return existingResult;
  }
  let promise = sendHttpRequest(openShardedHttpRequest(baseUrls, path + '/info'), 'json')
                    .then(response => new MultiscaleVolumeChunkSource(baseUrls, path, response));
  existingVolumes.set(fullKey, promise);
  return promise;
}

export function getVolume(url: string) {
  const [baseUrls, path] = parseSpecialUrl(url);
  return getShardedVolume(baseUrls, path);
}

registerDataSourceFactory('precomputed', {
  description: 'Precomputed file-backed data source',
  getVolume: getVolume,
  getMeshSource: getMeshSource,
});
