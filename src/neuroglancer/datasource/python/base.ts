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

export enum VolumeChunkEncoding {
  JPEG,
  NPZ,
  RAW
}

export interface VolumeChunkSourceParameters {
  baseUrls: string[];
  key: string;
  scaleKey: string;
  encoding: VolumeChunkEncoding;
}

export function volumeSourceToString(parameters: VolumeChunkSourceParameters) {
  return `python:volume:${parameters['baseUrls'][0]}/${parameters['key']}/${parameters['scaleKey']}/${VolumeChunkEncoding[parameters['encoding']]}`;
}

export interface MeshSourceParameters {
  baseUrls: string[];
  key: string;
}

export function meshSourceToString(parameters: MeshSourceParameters) {
  return `python:mesh:${parameters['baseUrls'][0]}/${parameters['key']}`;
}
