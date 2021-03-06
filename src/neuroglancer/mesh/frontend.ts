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

import {ChunkSourceParametersConstructor, ChunkState} from 'neuroglancer/chunk_manager/base';
import {Chunk, ChunkManager, ChunkSource} from 'neuroglancer/chunk_manager/frontend';
import {FRAGMENT_SOURCE_RPC_ID, MESH_LAYER_RPC_ID} from 'neuroglancer/mesh/base';
import {PerspectiveViewRenderContext, PerspectiveViewRenderLayer, perspectivePanelEmit} from 'neuroglancer/perspective_panel';
import {SegmentationDisplayState} from 'neuroglancer/segmentation_display_state';
import {Mat4, Vec3, mat4, vec3, vec4} from 'neuroglancer/util/geom';
import {stableStringify} from 'neuroglancer/util/json';
import {Buffer} from 'neuroglancer/webgl/buffer';
import {GL} from 'neuroglancer/webgl/context';
import {ShaderBuilder, ShaderProgram} from 'neuroglancer/webgl/shader';
import {setVec4FromUint32} from 'neuroglancer/webgl/shader_lib';
import {RPC, SharedObject} from 'neuroglancer/worker_rpc';

export class MeshShaderManager {
  private tempLightVec = vec4.create();
  private tempPickID = new Float32Array(4);
  constructor() {}

  defineShader(builder: ShaderBuilder) {
    builder.addAttribute('highp vec3', 'aVertexPosition');
    builder.addAttribute('highp vec3', 'aVertexNormal');
    builder.addVarying('highp vec3', 'vColor');
    builder.addUniform('highp vec4', 'uLightDirection');
    builder.addUniform('highp vec3', 'uColor');
    builder.addUniform('highp mat4', 'uModelMatrix');
    builder.addUniform('highp mat4', 'uProjection');
    builder.addUniform('highp vec4', 'uPickID');
    builder.require(perspectivePanelEmit);
    builder.setVertexMain(`
gl_Position = uProjection * (uModelMatrix * vec4(aVertexPosition, 1.0));
vec3 normal = (uModelMatrix * vec4(aVertexNormal, 0.0)).xyz;
float lightingFactor = abs(dot(normal, uLightDirection.xyz)) + uLightDirection.w;
vColor = lightingFactor * uColor;
`);
    builder.setFragmentMain(`emit(vec4(vColor, 1.0), uPickID);`);
  }

  beginLayer(gl: GL, shader: ShaderProgram, renderContext: PerspectiveViewRenderContext) {
    let {dataToDevice, lightDirection, ambientLighting, directionalLighting} = renderContext;
    gl.uniformMatrix4fv(shader.uniform('uProjection'), false, dataToDevice);
    let lightVec = this.tempLightVec;
    vec3.scale(lightVec, lightDirection, directionalLighting);
    lightVec[3] = ambientLighting;
    gl.uniform4fv(shader.uniform('uLightDirection'), lightVec);
  }

  beginObject(
      gl: GL, shader: ShaderProgram, objectToDataMatrix: Mat4, color: Vec3, pickID: number) {
    gl.uniformMatrix4fv(shader.uniform('uModelMatrix'), false, objectToDataMatrix);
    gl.uniform4fv(shader.uniform('uPickID'), setVec4FromUint32(this.tempPickID, pickID));
    gl.uniform3fv(shader.uniform('uColor'), color);
  }

  getShader(gl: GL) {
    return gl.memoize.get('mesh/MeshShaderManager', () => {
      let builder = new ShaderBuilder(gl);
      this.defineShader(builder);
      return builder.build();
    });
  }

  drawFragment(gl: GL, shader: ShaderProgram, fragmentChunk: FragmentChunk) {
    fragmentChunk.vertexBuffer.bindToVertexAttrib(
        shader.attribute('aVertexPosition'),
        /*components=*/3);

    fragmentChunk.normalBuffer.bindToVertexAttrib(
        shader.attribute('aVertexNormal'),
        /*components=*/3);
    fragmentChunk.indexBuffer.bind();
    gl.drawElements(gl.TRIANGLES, fragmentChunk.numIndices, gl.UNSIGNED_INT, 0);
  }
  endLayer(gl: GL, shader: ShaderProgram) {
    gl.disableVertexAttribArray(shader.attribute('aVertexPosition'));
    gl.disableVertexAttribArray(shader.attribute('aVertexNormal'));
  }
};

export class MeshLayer extends PerspectiveViewRenderLayer {
  private meshShaderManager = new MeshShaderManager();
  private shader = this.registerDisposer(this.meshShaderManager.getShader(this.gl));

  constructor(
      public chunkManager: ChunkManager, public source: MeshSource,
      public displayState: SegmentationDisplayState) {
    super();

    let dispatchRedrawNeeded = () => { this.redrawNeeded.dispatch(); };
    this.registerSignalBinding(displayState.segmentColorHash.changed.add(dispatchRedrawNeeded));
    this.registerSignalBinding(displayState.visibleSegments.changed.add(dispatchRedrawNeeded));
    this.registerSignalBinding(
        displayState.segmentSelectionState.changed.add(dispatchRedrawNeeded));

    let sharedObject = this.registerDisposer(new SharedObject());
    sharedObject.initializeCounterpart(chunkManager.rpc!, {
      'type': MESH_LAYER_RPC_ID,
      'chunkManager': chunkManager.rpcId,
      'source': source.addCounterpartRef(),
      'visibleSegmentSet': displayState.visibleSegments.rpcId
    });
    this.setReady(true);
  }

  get gl() { return this.chunkManager.chunkQueueManager.gl; }

  draw(renderContext: PerspectiveViewRenderContext) {
    let gl = this.gl;
    let shader = this.shader;
    shader.bind();
    let {meshShaderManager} = this;
    meshShaderManager.beginLayer(gl, shader, renderContext);

    let objectChunks = this.source.fragmentSource.objectChunks;

    let {pickIDs} = renderContext;

    // FIXME: this maybe should change
    let objectToDataMatrix = mat4.create();
    mat4.identity(objectToDataMatrix);

    let color = vec3.create();
    let {displayState} = this;
    let {segmentColorHash, segmentSelectionState} = displayState;

    for (let objectId of displayState.visibleSegments) {
      let objectKey = `${objectId.low}:${objectId.high}`;
      let fragments = objectChunks.get(objectKey);
      if (fragments === undefined) {
        continue;
      }
      segmentColorHash.compute(color, objectId);
      if (segmentSelectionState.isSelected(objectId)) {
        for (let i = 0; i < 3; ++i) {
          color[i] = color[i] * 0.5 + 0.5;
        }
      }
      meshShaderManager.beginObject(
          gl, shader, objectToDataMatrix, color, pickIDs.register(this, objectId));
      for (let fragment of fragments) {
        if (fragment.state === ChunkState.GPU_MEMORY) {
          meshShaderManager.drawFragment(gl, shader, fragment);
        }
      }
    }

    meshShaderManager.endLayer(gl, shader);
  }
};

export class FragmentChunk extends Chunk {
  vertexPositions: Float32Array;
  indices: Uint32Array;
  vertexNormals: Float32Array;
  objectKey: string;
  source: FragmentSource;
  vertexBuffer: Buffer;
  indexBuffer: Buffer;
  normalBuffer: Buffer;
  numIndices: number;

  constructor(source: FragmentSource, x: any) {
    super(source);
    this.objectKey = x['objectKey'];
    this.vertexPositions = x['vertexPositions'];
    let indices = this.indices = x['indices'];
    this.numIndices = indices.length;
    this.vertexNormals = x['vertexNormals'];
  }

  copyToGPU(gl: GL) {
    super.copyToGPU(gl);
    this.vertexBuffer = Buffer.fromData(gl, this.vertexPositions, gl.ARRAY_BUFFER, gl.STATIC_DRAW);
    this.indexBuffer = Buffer.fromData(gl, this.indices, gl.ELEMENT_ARRAY_BUFFER, gl.STATIC_DRAW);
    this.normalBuffer = Buffer.fromData(gl, this.vertexNormals, gl.ARRAY_BUFFER, gl.STATIC_DRAW);
  }

  freeGPUMemory(gl: GL) {
    super.freeGPUMemory(gl);
    this.vertexBuffer.dispose();
    this.indexBuffer.dispose();
    this.normalBuffer.dispose();
  }
};

export class FragmentSource extends ChunkSource {
  objectChunks = new Map<string, Set<FragmentChunk>>();
  constructor(chunkManager: ChunkManager, public meshSource: MeshSource) {
    super(chunkManager);
    this.initializeCounterpart(chunkManager.rpc!, {'type': FRAGMENT_SOURCE_RPC_ID});
  }
  addChunk(key: string, chunk: FragmentChunk) {
    super.addChunk(key, chunk);
    let {objectChunks} = this;
    let {objectKey} = chunk;
    let fragments = objectChunks.get(objectKey);
    if (fragments === undefined) {
      fragments = new Set();
      objectChunks.set(objectKey, fragments);
    }
    fragments.add(chunk);
  }
  deleteChunk(key: string) {
    let chunk = <FragmentChunk>this.chunks.get(key);
    super.deleteChunk(key);
    let {objectChunks} = this;
    let {objectKey} = chunk;
    let fragments = objectChunks.get(objectKey)!;
    fragments.delete(chunk);
    if (fragments.size === 0) {
      objectChunks.delete(objectKey);
    }
  }

  getChunk(x: any) { return new FragmentChunk(this, x); }
};

export abstract class MeshSource extends ChunkSource {
  fragmentSource = new FragmentSource(this.chunkManager, this);
  initializeCounterpart(rpc: RPC, options: any) {
    options['fragmentSource'] = this.fragmentSource.addCounterpartRef();
    super.initializeCounterpart(rpc, options);
  }
};

/**
 * Defines a MeshSource for which all state is encapsulated in an object of type Parameters.
 */
export function defineParameterizedMeshSource<Parameters>(
    parametersConstructor: ChunkSourceParametersConstructor<Parameters>) {
  return class ParameterizedMeshSource extends MeshSource {
    constructor(chunkManager: ChunkManager, public parameters: Parameters) {
      super(chunkManager);
      this.initializeCounterpart(
          chunkManager.rpc!, {'type': parametersConstructor.RPC_ID, 'parameters': parameters});
    }
    static get(chunkManager: ChunkManager, parameters: Parameters) {
      return chunkManager.getChunkSource(
          this, stableStringify(parameters), () => new this(chunkManager, parameters));
    }
    toString() { return parametersConstructor.stringify(this.parameters); }
  };
}
