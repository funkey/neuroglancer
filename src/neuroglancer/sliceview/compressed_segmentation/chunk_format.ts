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

import {DataType, VolumeChunkSpecification} from 'neuroglancer/sliceview/base';
import {readSingleChannelValue as readSingleChannelValueUint32} from 'neuroglancer/sliceview/compressed_segmentation/decode_uint32';
import {readSingleChannelValue as readSingleChannelValueUint64} from 'neuroglancer/sliceview/compressed_segmentation/decode_uint64';
import {VolumeChunkSource, ChunkFormatHandler, registerChunkFormatHandler} from 'neuroglancer/sliceview/frontend';
import {GLSL_TYPE_FOR_DATA_TYPE} from 'neuroglancer/sliceview/renderlayer';
import {SingleTextureChunkFormat, SingleTextureVolumeChunk} from 'neuroglancer/sliceview/single_texture_chunk_format';
import {maybePadArray} from 'neuroglancer/util/array';
import {RefCounted} from 'neuroglancer/util/disposable';
import {vec3, Vec3, vec3Key} from 'neuroglancer/util/geom';
import {Uint64} from 'neuroglancer/util/uint64';
import {GL} from 'neuroglancer/webgl/context';
import {ShaderBuilder, ShaderProgram} from 'neuroglancer/webgl/shader';
import {glsl_getFortranOrderIndexFromNormalized, glsl_uint32, glsl_uint64} from 'neuroglancer/webgl/shader_lib';

class TextureLayout extends RefCounted {
  textureWidth: number;
  textureHeight: number;
  textureAccessCoefficients: Float32Array;
  subchunkGridSize: Vec3;
  channelOffsets: Float32Array;

  constructor(gl: GL, public chunkDataSize: Vec3, public subchunkSize: Vec3, dataLength: number) {
    super();
    let {maxTextureSize} = gl;

    // Use arbitrary layout.
    let dataWidth = Math.ceil(dataLength / maxTextureSize);
    if (dataWidth > maxTextureSize) {
      throw new Error('Chunk data size exceeds maximum texture size: ' + dataLength);
    }

    let dataHeight = Math.ceil(dataLength / dataWidth);
    this.textureWidth = dataWidth;
    this.textureHeight = dataHeight;
    this.textureAccessCoefficients =
        Float32Array.of(1.0 / dataWidth, 1.0 / (dataWidth * dataHeight));
    let subchunkGridSize = this.subchunkGridSize = vec3.create();
    for (let i = 0; i < 3; ++i) {
      subchunkGridSize[i] = Math.ceil(chunkDataSize[i] / subchunkSize[i]);
    }
  }

  static get(gl: GL, chunkDataSize: Vec3, subchunkSize: Vec3, dataLength: number) {
    return gl.memoize.get(
        `sliceview.CompressedSegmentationTextureLayout:${vec3Key(chunkDataSize)},${vec3Key(subchunkSize)},${dataLength}`,
        () => new TextureLayout(gl, chunkDataSize, subchunkSize, dataLength));
  }
};

class ChunkFormat extends SingleTextureChunkFormat<TextureLayout> {
  static get(gl: GL, dataType: DataType, subchunkSize: Vec3, numChannels: number) {
    let key =
        `sliceview.CompressedSegmentationChunkFormat:${dataType}:${vec3Key(subchunkSize)}:${numChannels}`;
    return gl.memoize.get(key, () => new ChunkFormat(dataType, subchunkSize, numChannels, key));
  }

  constructor(
      public dataType: DataType, public subchunkSize: Vec3, public numChannels: number,
      key: string) {
    super(key);
  }

  defineShader(builder: ShaderBuilder) {
    super.defineShader(builder);
    let local = (x: string) => 'compressedSegmentationChunkFormat_' + x;
    builder.addUniform('highp vec2', 'uCompressedSegmentationTextureAccessCoefficients');
    builder.addUniform('highp vec3', 'uSubchunkGridSize');
    builder.addUniform('highp vec3', 'uChunkDataSize');
    builder.addFragmentCode(glsl_getFortranOrderIndexFromNormalized);
    const {dataType} = this;
    const glslType = GLSL_TYPE_FOR_DATA_TYPE.get(dataType);

    if (dataType === DataType.UINT64) {
      builder.addFragmentCode(glsl_uint64);
    } else {
      builder.addFragmentCode(glsl_uint32);
    }

    let fragmentCode = `
vec4 ${local('readTextureValue')}(float offset) {
  // We add 0.5 to avoid being right at a texel boundary.
  offset += 0.5;
  return texture2D(uVolumeChunkSampler,
                   vec2(fract(offset * uCompressedSegmentationTextureAccessCoefficients.x),
                        offset * uCompressedSegmentationTextureAccessCoefficients.y));
}
float ${local('getChannelOffset')}(int channelIndex) {
  if (channelIndex == 0) {
    return 1.0;
  }
  vec4 v = ${local('readTextureValue')}(float(channelIndex));
  return v.x * 255.0 + (v.y * 255.0) * 256.0 + (v.z * 255.0) * 256.0 * 256.0;
}
${glslType} getDataValue (int channelIndex) {
  const vec3 uSubchunkSize = ${vec3.str(this.subchunkSize)};

  vec3 chunkPosition = getSubscriptsFromNormalized(vChunkPosition, uChunkDataSize);

  // TODO: maybe premultiply this and store as uniform.
  vec3 subchunkGridPosition = floor(chunkPosition / uSubchunkSize);
  float subchunkGridOffset = getFortranOrderIndex(subchunkGridPosition, uSubchunkGridSize);

  float channelOffset = ${local('getChannelOffset')}(channelIndex);

  // TODO: Maybe just combine this offset into subchunkGridStrides.
  float subchunkHeaderOffset = subchunkGridOffset * 2.0 + channelOffset;

  vec4 subchunkHeader0 = ${local('readTextureValue')}(subchunkHeaderOffset);
  vec4 subchunkHeader1 = ${local('readTextureValue')}(subchunkHeaderOffset + 1.0);

  float outputValueOffset = dot(subchunkHeader0.xyz, vec3(255, 256 * 255, 256 * 256 * 255)) + channelOffset;
  float encodingBits = subchunkHeader0[3] * 255.0;
  if (encodingBits > 0.0) {
    vec3 subchunkPosition = floor(min(chunkPosition - subchunkGridPosition * uSubchunkSize, uSubchunkSize - 1.0));
    float subchunkOffset = getFortranOrderIndex(subchunkPosition, uSubchunkSize);
    highp float encodedValueBaseOffset = dot(subchunkHeader1.xyz, vec3(255.0, 256.0 * 255.0, 256.0 * 256.0 * 255.0)) + channelOffset;
    highp float encodedValueOffset = floor(encodedValueBaseOffset + subchunkOffset * encodingBits / 32.0);
    vec4 encodedValue = ${local('readTextureValue')}(encodedValueOffset);
    float wordOffset = mod(subchunkOffset * encodingBits, 32.0);
    // If the value is in the first byte, then 0 <= wordOffset < 8.
    // We need to mod by 2**encodedBits
    float wordShifter = pow(2.0, -wordOffset);
    float encodedValueMod = pow(2.0, encodingBits);
    float encodedValueShifted;
    if (wordOffset < 16.0) {
      encodedValueShifted = dot(encodedValue.xy, vec2(255.0, 255.0 * 256.0));
    } else {
      encodedValueShifted = dot(encodedValue.zw, vec2(255.0 * 256.0 * 256.0, 255.0 * 256.0 * 256.0 * 256.0));
    }
    encodedValueShifted = floor(encodedValueShifted * wordShifter);
    float decodedValue = mod(encodedValueShifted, encodedValueMod);
    outputValueOffset += decodedValue * ${this.dataType === DataType.UINT64 ? '2.0' : '1.0'};
  }
  ${glslType} result;
`;
    if (dataType === DataType.UINT64) {
      fragmentCode += `
  result.low = ${local('readTextureValue')}(outputValueOffset);
  result.high = ${local('readTextureValue')}(outputValueOffset+1.0);
`;
    } else {
      fragmentCode += `
  result.value = ${local('readTextureValue')}(outputValueOffset);
`;
    }
    fragmentCode += `
  return result;
}
`;
    builder.addFragmentCode(fragmentCode);
  }

  /**
   * Called each time textureLayout changes while drawing chunks.
   */
  setupTextureLayout(gl: GL, shader: ShaderProgram, textureLayout: TextureLayout) {
    gl.uniform3fv(shader.uniform('uChunkDataSize'), textureLayout.chunkDataSize);
    gl.uniform3fv(shader.uniform('uSubchunkGridSize'), textureLayout.subchunkGridSize);
    gl.uniform2fv(
        shader.uniform('uCompressedSegmentationTextureAccessCoefficients'),
        textureLayout.textureAccessCoefficients);
  }

  getTextureLayout(gl: GL, chunkDataSize: Vec3, dataLength: number) {
    return TextureLayout.get(gl, chunkDataSize, this.subchunkSize, dataLength);
  }
};

export class CompressedSegmentationVolumeChunk extends
    SingleTextureVolumeChunk<Uint32Array, TextureLayout> {
  chunkFormat: ChunkFormat;

  setTextureData(gl: GL) {
    let {data} = this;
    let {chunkFormat} = this;
    let textureLayout = this.textureLayout =
        chunkFormat.getTextureLayout(gl, this.chunkDataSize, data.length);
    let requiredSize = textureLayout.textureWidth * textureLayout.textureHeight;
    let padded = maybePadArray(data, requiredSize);
    gl.texImage2D(
        gl.TEXTURE_2D,
        /*level=*/0, gl.RGBA,
        /*width=*/textureLayout.textureWidth,
        /*height=*/textureLayout.textureHeight,
        /*border=*/0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array(padded.buffer, padded.byteOffset, padded.byteLength));
  }

  getChannelValueAt(dataPosition: Vec3, channel: number): Uint64|number {
    let {chunkDataSize, chunkFormat} = this;
    let {data} = this;
    let offset = data[channel];
    if (chunkFormat.dataType === DataType.UINT64) {
      let result = new Uint64();
      readSingleChannelValueUint64(
          result, data, /*baseOffset=*/offset, chunkDataSize, chunkFormat.subchunkSize,
        dataPosition);
      return result;
    } else {
      return readSingleChannelValueUint32(
          data, /*baseOffset=*/offset, chunkDataSize, chunkFormat.subchunkSize, dataPosition);
    }
  }
};

export class CompressedSegmentationChunkFormatHandler extends RefCounted implements
    ChunkFormatHandler {
  chunkFormat: ChunkFormat;

  constructor(gl: GL, spec: VolumeChunkSpecification) {
    super();
    let {dataType} = spec;
    if (dataType !== DataType.UINT64 && dataType !== DataType.UINT32) {
      throw new Error(`Unsupported compressed segmentation data type: ${DataType[dataType]}`);
    }
    this.chunkFormat = this.registerDisposer(
        ChunkFormat.get(gl, spec.dataType, spec.compressedSegmentationBlockSize, spec.numChannels));
  }

  getChunk(source: VolumeChunkSource, x: any) {
    return new CompressedSegmentationVolumeChunk(source, x);
  }
};

registerChunkFormatHandler((gl: GL, spec: VolumeChunkSpecification) => {
  if (spec.compressedSegmentationBlockSize != null) {
    return new CompressedSegmentationChunkFormatHandler(gl, spec);
  }
  return null;
});
