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

import {handleChunkDownloadPromise} from 'neuroglancer/chunk_manager/backend';
import {MeshSourceParameters, VolumeChunkEncoding, VolumeChunkSourceParameters, meshSourceToString, volumeSourceToString} from 'neuroglancer/datasource/python/base';
import {FragmentChunk, ManifestChunk, MeshSource as GenericMeshSource, decodeVertexPositionsAndIndices} from 'neuroglancer/mesh/backend';
import {VolumeChunk, VolumeChunkSource as GenericVolumeChunkSource} from 'neuroglancer/sliceview/backend';
import {ChunkDecoder} from 'neuroglancer/sliceview/backend_chunk_decoders';
import {decodeJpegChunk} from 'neuroglancer/sliceview/backend_chunk_decoders/jpeg';
import {decodeNdstoreNpzChunk} from 'neuroglancer/sliceview/backend_chunk_decoders/ndstoreNpz';
import {decodeRawChunk} from 'neuroglancer/sliceview/backend_chunk_decoders/raw';
import {Endianness} from 'neuroglancer/util/endian';
import {openShardedHttpRequest, sendHttpRequest} from 'neuroglancer/util/http_request';
import {RPC, registerSharedObject} from 'neuroglancer/worker_rpc';

let chunkDecoders = new Map<VolumeChunkEncoding, ChunkDecoder>();
chunkDecoders.set(VolumeChunkEncoding.NPZ, decodeNdstoreNpzChunk);
chunkDecoders.set(VolumeChunkEncoding.JPEG, decodeJpegChunk);
chunkDecoders.set(VolumeChunkEncoding.RAW, decodeRawChunk);

class VolumeChunkSource extends GenericVolumeChunkSource {
  chunkDecoder: ChunkDecoder;
  parameters: VolumeChunkSourceParameters;
  encoding: string;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.parameters = options['parameters'];
    this.chunkDecoder = chunkDecoders.get(this.parameters['encoding'])!;
    this.encoding = VolumeChunkEncoding[this.parameters.encoding].toLowerCase();
  }

  download(chunk: VolumeChunk) {
    let {parameters} = this;
    console.log(parameters);
    let path = `/neuroglancer/${this.encoding}/${parameters.key}/${parameters.scaleKey}`;
    {
      // chunkPosition must not be captured, since it will be invalidated by the next call to
      // computeChunkBounds.
      let chunkPosition = this.computeChunkBounds(chunk);
      let {chunkDataSize} = chunk;
      for (let i = 0; i < 3; ++i) {
        path += `/${chunkPosition[i]},${chunkPosition[i] + chunkDataSize![i]}`;
      }
    }
    handleChunkDownloadPromise(
        chunk, sendHttpRequest(openShardedHttpRequest(parameters.baseUrls, path), 'arraybuffer'),
        this.chunkDecoder);
  }

  toString() { return volumeSourceToString(this.parameters); }
};
registerSharedObject('python/VolumeChunkSource', VolumeChunkSource);

export function decodeFragmentChunk(chunk: FragmentChunk, response: ArrayBuffer) {
  let dv = new DataView(response);
  let numVertices = dv.getUint32(0, true);
  decodeVertexPositionsAndIndices(
      chunk, response, Endianness.LITTLE, /*vertexByteOffset=*/4, numVertices);
}

function decodeManifestChunk(chunk: ManifestChunk, response: any) {
  chunk.fragmentIds = [''];
}

export class MeshSource extends GenericMeshSource {
  parameters: MeshSourceParameters;

  constructor(rpc: RPC, options: any) {
    super(rpc, options);
    this.parameters = options['parameters'];
  }

  download(chunk: ManifestChunk) {
    // No manifest chunk to download, as there is always only a single fragment.
    handleChunkDownloadPromise(chunk, Promise.resolve(undefined), decodeManifestChunk);
  }

  downloadFragment(chunk: FragmentChunk) {
    let {parameters} = this;
    let requestPath = `/neuroglancer/mesh/${parameters.key}/${chunk.manifestChunk!.objectId}`;
    handleChunkDownloadPromise(
        chunk,
        sendHttpRequest(openShardedHttpRequest(parameters.baseUrls, requestPath), 'arraybuffer'),
        decodeFragmentChunk);
  }
  toString() { return meshSourceToString(this.parameters); }
};
registerSharedObject('python/MeshSource', MeshSource);
